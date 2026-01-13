import numpy as np
import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from utils import MIMICPipeline, MIMICDataset
from fsl import MIMICPredictor
import torch.nn.functional as F

def mimic_loss(pred, target, mask):
    """
    Composite Loss to avoid collapse to the mean.
    """
    # Reconstruction (L1) only on masked parts
    l1_loss = F.l1_loss(pred * mask, target * mask, reduction='sum') / (torch.sum(mask) + 1e-8) # for numerical stability
    
    # Shape - gradient loss
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    target_diff = target[:, 1:, :] - target[:, :-1, :]
    mask_diff = mask[:, 1:, :] * mask[:, :-1, :]
    
    shape_loss = F.l1_loss(pred_diff * mask_diff, target_diff * mask_diff, reduction='sum') / (torch.sum(mask_diff) + 1e-8)
    
    # Variance Loss
    # Forces the model to produce structured "noise" rather than a flat line.
    # We calculate the standard deviation along the temporal axis (dim=1)
    std_pred = torch.std(pred, dim=1)
    std_target = torch.std(target, dim=1)
    
    # We want the predicted variance to be close to the true variance
    var_loss = F.mse_loss(std_pred, std_target)
    
    # Weighting
    return l1_loss + 1.5 * shape_loss + 0.3 * var_loss

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
    
    # Split train/val on patients
    all_patients = df.index.get_level_values('hadm_id').unique().to_numpy()
    np.random.shuffle(all_patients)
    
    split_idx = int(len(all_patients) * 0.8)
    train_ids = all_patients[:split_idx]
    val_ids = all_patients[split_idx:]
    
    print(f"Split Patients: {len(train_ids)} Train / {len(val_ids)} Val")
    
    # MultiIndex slicing
    df_train = df[df.index.get_level_values('hadm_id').isin(train_ids)]
    df_val = df[df.index.get_level_values('hadm_id').isin(val_ids)]
    
    # Dataset creation
    train_dataset = MIMICDataset(df_train, window_size=24, mask_ratio=0.30) # change to 30% mask ratio (from #Alpha 1.0.1 : 50%)
    train_stats = train_dataset.get_stats()

    val_dataset = MIMICDataset(df_val, window_size=24, mask_ratio=0.30, stats=train_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    
    # Configuration
    n_features = df.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Model initialization for {n_features} vital signs...")
    
    model = MIMICPredictor(n_assets=n_features, window_size=24).to(device)
    
    # Training configurations
    n_epochs = 30
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # try AdamW instead of Adam
    
    # Scheduler to reduce LR on plateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print("Starting training...")
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        # Training Loop
        for inputs, targets, mask in train_loader:
            inputs = [x.to(device) for x in inputs]
            targets = targets.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            _, res = model(inputs)
            reconstructed = torch.stack(res['outputs'], dim=2)
            
            # Mimic Loss (check details in the function)
            loss = mimic_loss(reconstructed, targets, mask)
            
            # Adding H¹ with small weight at the start (avoiding collapse)
            h1_loss = res['h1_score']
            total_objective = loss + 0.05 * h1_loss
            
            total_objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Loop 
        model.eval()
        val_loss_acc = 0
        with torch.no_grad():
            for inputs, targets, mask in val_loader:
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
                mask = mask.to(device)
                
                _, res = model(inputs)
                reconstructed = torch.stack(res['outputs'], dim=2)
                val_loss_acc += mimic_loss(reconstructed, targets, mask).item()
        
        avg_val_loss = val_loss_acc / len(val_loader)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
    
    # Save the model
    torch.save(model.state_dict(), "./saves/mimic_fsl.pth")
    print("Model saved.")

if __name__ == "__main__":
    train_mimic_reconstruction()