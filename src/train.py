import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from utils import MIMICPipeline, MIMICDataset
from fsl import MIMICPredictor

def sharp_loss(pred, target, mask):
    """
    Combines L1 (Value) and Diff (Slope) to avoid flat lines.
    """
    # Classic L1 Loss (Absolute Value)
    # Punishes error without being too sensitive to outliers (unlike squared)
    l1 = torch.abs(pred - target) * mask
    term_val = torch.sum(l1) / (torch.sum(mask) + 1e-8)
    
    # Derivative Loss (Shape Penalty)
    # Forces the predicted slope (t+1 - t) to resemble the true slope
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]       # Predicted slope
    target_diff = target[:, 1:, :] - target[:, :-1, :] # True slope
    
    # Adjust the mask because we lost 1 time point with the diff
    mask_diff = mask[:, 1:, :] * mask[:, :-1, :] 
    
    l1_diff = torch.abs(pred_diff - target_diff) * mask_diff
    term_shape = torch.sum(l1_diff) / (torch.sum(mask_diff) + 1e-8)
    
    # We combine: 1.0 * Value + 1.0 * Shape
    return term_val + 1.0 * term_shape

def train_mimic_reconstruction():
    # Load data
    pipeline = MIMICPipeline()
    try:
        if os.path.exists("./saves/cache_mimic.pkl"):
            import pandas as pd
            df = pd.read_pickle("./saves/cache_mimic.pkl")
            print("Cache loaded.")
        else:
            df = pipeline.load_cohort(n_patients=200)
    except Exception as e:
        print(f"Data error: {e}")
        return

    # Dataset
    dataset = MIMICDataset(df, window_size=24, mask_ratio=0.20)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Configuration
    n_features = df.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Initialisation du modèle pour {n_features} signes vitaux...")
    model = MIMICPredictor(n_assets=n_features, window_size=24).to(device)
    
    # Training configurations
    n_epochs = 30
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Scheduler to reduce LR on plateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print("Starting training...")
    
    for epoch in range(n_epochs):
        total_loss = 0
        total_h1 = 0
        
        model.train()
        
        for inputs, targets, mask in loader:
            inputs = [x.to(device) for x in inputs]
            targets = targets.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            _, res = model(inputs)
            
            # Loss computation :
            reconstructed_stack = torch.stack(res['outputs'], dim=2)
            
            # Use our new sharp_loss function
            reconstruction_loss = sharp_loss(reconstructed_stack, targets, mask)
            
            # Retrieve the H1
            h1_loss = res['h1_score']
            
            # Total (We weight the H1)
            loss = reconstruction_loss + 0.1 * h1_loss
            
            loss.backward()
            
            # Gradient Clipping
            # Prevent the model from making too large jumps if the Loss explodes
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Logs
            total_loss += reconstruction_loss.item()
            total_h1 += h1_loss.item() if isinstance(h1_loss, torch.Tensor) else h1_loss
            
        # Averages for the epoch
        avg_loss = total_loss / len(loader)
        avg_h1 = total_h1 / len(loader)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Loss (Sharp): {avg_loss:.4f} | H1: {avg_h1:.6f}")
        
        scheduler.step(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")
    
    # Save the model
    torch.save(model.state_dict(), "./saves/mimic_fsl.pth")
    print("Model saved.")

if __name__ == "__main__":
    train_mimic_reconstruction()