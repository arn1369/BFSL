import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

from fsl import HierarchicalFSL, TopologicalTripletLoss

def get_universal_data():
    """
    Downloads and cleans S&P 500 data.
    Returns RAW RETURNS.
    Global normalization is avoided here to prevent Look-Ahead Bias.
    """
    
    print("Downloading Universal Dataset...")
    
    # Selection of diverse sectors (Tech, Finance, Energy, Health, etc.)
    # to ensure the model learns universal market dynamics.
    tickers = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", 
        "JPM", "BAC", 
        "XOM", "CVX",
        "JNJ", "PFE",
        "PG", "KO", "MCD", "HD",
        "DIS"
    ]
    
    data = yf.download(tickers, start="2010-01-01", end="2018-12-31", progress=True, auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if 'Close' in data.columns.get_level_values(0):
                df = data['Close']
            else:
                df = data.xs('Close', axis=1, level=0)
        except:
            print("This isn't supposed to happen.")
            df = data['Adj Close']
    else:
        df = data['Close'] if 'Close' in data.columns else data

    # Cleaning Data
    df = df.dropna(axis=1, thresh=int(len(df)*0.9)) # keep tickers with at least 90% data
    df = df.ffill().bfill()
    
    returns = df.pct_change().dropna()
    
    print(f"Valid tickers retrieved: {len(returns.columns)}")
    print(f"Data ready. Shape: {returns.shape}")
    
    return returns, returns.columns.tolist()

class UniversalDataset(Dataset):
    def __init__(self, dataframe, window_size=20, augment=False):
        """
        dataframe : DataFrame of RAW returns.
        """
        self.data = dataframe
        self.window_size = window_size
        self.augment = augment
        self.n_assets = dataframe.shape[1]
        self.values = dataframe.values 

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # Extraction : Get the window and the next value (target)
        raw_window = self.values[idx : idx + self.window_size]
        target = self.values[idx + self.window_size]
        
        # Instance Normalization: Normalize each window independently (Mean=0, Std=1)
        # This is crucial for financial data to make it comparable across different volatility regimes.
        window_mean = np.mean(raw_window, axis=0)
        window_std = np.std(raw_window, axis=0) + 1e-8
        normalized_window = (raw_window - window_mean) / window_std
        
        # Augmentation: Add noise during training
        if self.augment:
            noise = np.random.normal(0, 0.05, normalized_window.shape)
            normalized_window = normalized_window + noise

        # Tensorization
        inputs = []
        for asset_idx in range(self.n_assets):
            asset_series = normalized_window[:, asset_idx]
            
            tensor = torch.tensor(asset_series, dtype=torch.float) 
            
            inputs.append(tensor)
            
        target_tensor = torch.tensor(target, dtype=torch.float)

        return inputs, target_tensor

class FSLPredictor(nn.Module):
    """
    Wraps the Hierarchical FSL backbone with a prediction head.
    The FSL extracts features, and the head predicts the next return.
    """
    def __init__(self, n_assets, window_size):
        super().__init__()
        self.fsl = HierarchicalFSL(
            scales=[16, 4, 1], # Hierarchical reduction
            context_dim=window_size,
            attention_dim=32,
            diffusion_steps=[1, 2, 2]
        )
        # Simple linear head for forecasting
        self.head = nn.Sequential(
            nn.Linear(window_size, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_list):
        res = self.fsl(x_list)
        clean_sections = res['sections']
        
        # Predict separately for each asset using the shared head
        preds = []
        for section in clean_sections:
            preds.append(self.head(section))
            
        return torch.cat(preds, dim=1), res

def correlation_loss(pred, target):
    """
    Maximizes Pearson correlation between prediction and target.
    Encourages the model to get the relative ranking of assets right.
    """
    pred_n = (pred - pred.mean(dim=1, keepdim=True)) / (pred.std(dim=1, keepdim=True) + 1e-8)
    target_n = (target - target.mean(dim=1, keepdim=True)) / (target.std(dim=1, keepdim=True) + 1e-8)
    corr = (pred_n * target_n).mean(dim=1)
    return 1 - corr.mean()

def sign_loss(pred, target):
    """
    Penalizes directional errors.
    Uses a soft differentiable approximation (tanh) to penalize wrong signs.
    """
    # Scale by 5.0 to make the tanh slope steeper around 0
    pred_sign = torch.tanh(pred * 5.0)
    target_sign = torch.sign(target)
    
    # We want pred_sign and target_sign to match (product close to 1)
    return 1 - torch.mean(pred_sign * target_sign)

def train_model():
    # Load data
    try:
        df_univ, tickers = get_universal_data()
    except Exception as e:
        print(f"Critical error during download: {e}")
        return
    
    # Time-based split (Train: first 80%, Val: next 10%)
    train_split = int(len(df_univ) * 0.8)
    val_split = int(len(df_univ) * 0.9)
    
    train_df = df_univ.iloc[:train_split]
    val_df = df_univ.iloc[train_split:val_split]
    
    print(f"Train: {len(train_df)} days, Val: {len(val_df)} days")
    
    # Initialize Datasets (augment only on training)
    train_ds = UniversalDataset(train_df, augment=True)
    val_ds = UniversalDataset(val_df, augment=False)
    
    if len(train_ds) == 0:
        print("Error : Empty dataset.")
        return

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # Setup Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}.")
    
    os.makedirs("./saves", exist_ok=True)
    
    model = FSLPredictor(n_assets=16, window_size=20).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    # Initialize Losses
    triplet_criterion = TopologicalTripletLoss(margin=0.5, structural_weight=0.5).to(device)
    mse_criterion = nn.MSELoss()
    
    # Early Stopping parameters
    epochs = 20
    patience = 4
    best_val_loss = float('inf')
    patience_counter = 0
    
    model.train()
    
    for epoch in range(epochs):
        # TRAINING
        total_loss = 0
        batch_count = 0
        
        for inputs, targets in train_loader:
            inputs = [x.to(device).float() for x in inputs]
            targets = targets.to(device).float()
            
            # Anchor Pass (Clean Data)
            optimizer.zero_grad()
            preds_anchor, res_anchor = model(inputs)
            
            # Positive Pass (Slightly Noisy Data)
            # We want the topology (H1 score) to remain stable despite noise
            inputs_pos = [x + torch.randn_like(x) * 0.005 for x in inputs]
            _, res_pos = model(inputs_pos)
            
            # Negative Pass (Incoherent/Shuffled Data)
            # We want the topology to look very different (high H1) for shuffled data
            batch_size = inputs[0].size(0)
            inputs_neg = []
            for x in inputs:
                perm = torch.randperm(batch_size)
                inputs_neg.append(x[perm])
            
            _, res_neg = model(inputs_neg)
                        
            #ANCHOR: LOSS COMPUTING
            
            # Prediction Accuracy (MSE)
            mse = mse_criterion(preds_anchor, targets)
            
            
            # Topological Triplet Loss
            # Objectives: Low H1 for Anchor/Pos, High H1 for Neg, Anchor ≈ Pos
            triplet_loss = triplet_criterion(
                h1_anchor=res_anchor['h1_score'],
                h1_positive=res_pos['h1_score'],
                h1_negative=res_neg['h1_score']
            )
            
            # Orthogonality & Reconstruction (Regularization)
            recon_loss = 0
            for original, reconstructed in zip(inputs, res_anchor['outputs']):
                recon_loss += torch.mean((original - reconstructed) ** 2)

            # Financial Metrics (Correlation & Sign)
            corr = correlation_loss(preds_anchor, targets)
            s_loss = sign_loss(preds_anchor, targets)
            
            # Total Loss combination
            loss = 1.0 * corr + 0.5 * s_loss + 0.1 * mse + 0.5 * triplet_loss + 0.1 * recon_loss
                    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = total_loss / batch_count
        
        
        # VALIDATION
        model.eval()
        val_loss = 0
        val_count = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = [x.to(device).float() for x in inputs]
                targets = targets.to(device).float()
                
                preds, res = model(inputs)
                
                # Validation metric: simplified loss
                mse = mse_criterion(preds, targets)
                recon_loss = 0
                for original, reconstructed in zip(inputs, res['outputs']):
                    recon_loss += torch.mean((original - reconstructed) ** 2)
                
                loss = mse + 0.5 * recon_loss
                val_loss += loss.item()
                val_count += 1
        
        avg_val_loss = val_loss / val_count
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        #ANCHOR: Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "./saves/fsl.pth")
            print(f"New best model saved (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        model.train()
        
    # Final save
    print("\nLoading best model for final save...")
    try:
        model.load_state_dict(torch.load("./saves/fsl.pth"))
    except FileNotFoundError:
        print("Warning: Best model not found, saving current state.")
    
    torch.save(model.state_dict(), "./saves/final_fsl.pth")
    print("Done.")

if __name__ == "__main__":
    torch.manual_seed(1) # reproducibility
    np.random.seed(1)
    train_model()