import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import MIMICPipeline, MIMICDataset, VITAL_IDS
from fsl import MIMICPredictor

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    WINDOW_SIZE = 24
    MASK_RATIO = 0.2
    
    print("Starting test...")
    
    # here need to load other patients instead of training ones
    pipeline = MIMICPipeline()
    try:
        #ANCHOR: Need to split dataset before !
        import os
        if os.path.exists("./saves/cache_mimic.pkl"):
            df = pd.read_pickle("./saves/cache_mimic.pkl")
            split_idx = int(len(df) * 0.8)
            #ANCHOR: Need to split per patient !
            df_test = df.iloc[split_idx:]
            print(f"Loaded Test set (from cache) : {len(df_test)} lines.")
        else:
            print("Cache not found. Please run train.py first.")
            return
    except Exception as e:
        print(f"Data error: {e}")
        return

    test_ds = MIMICDataset(df_test, window_size=WINDOW_SIZE, mask_ratio=MASK_RATIO)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    n_features = 6 # HR, SysBP, DiaBP, SpO2, Resp, Temp
    model = MIMICPredictor(n_assets=n_features, window_size=WINDOW_SIZE).to(device)
    
    try:
        model.load_state_dict(torch.load("./saves/mimic_fsl.pth", map_location=device))
        print("Model weights loaded.")
    except FileNotFoundError:
        print("Error: .pth file not found.")
        return

    model.eval()
    
    # Evaluation loop
    total_mae = 0
    total_mse = 0
    
    # To store errors per organ
    # Shape: (6,)
    per_feature_mae = np.zeros(n_features)
    per_feature_count = np.zeros(n_features)
    
    print(f"Evaluation on {len(test_ds)} windows...")
    
    with torch.no_grad():
        for inputs, targets, mask in test_loader:
            inputs = [x.to(device) for x in inputs]
            targets = targets.to(device)
            mask = mask.to(device)
            
            # Forward
            _, res = model(inputs)
            
            # Reconstruction
            reconstructed_stack = torch.stack(res['outputs'], dim=2) # (Batch, Time, Feats)
            
            # We only calculate the error on what was masked
            # Absolute Error (|Y_pred - Y_true|)
            abs_error = torch.abs(reconstructed_stack - targets) * mask
            
            # Update global metrics
            # We divide by the number of masked elements to get a true average
            n_masked = torch.sum(mask) + 1e-8
            
            batch_mae = torch.sum(abs_error) / n_masked
            batch_mse = torch.sum((reconstructed_stack - targets)**2 * mask) / n_masked
            
            total_mae += batch_mae.item()
            total_mse += batch_mse.item()
            
            # Update metrics per organ
            # We sum the errors for each feature (dim=2) over the entire batch and time
            # mask shape: (Batch, Time, Feats)
            for i in range(n_features):
                feat_mask = mask[:, :, i]
                feat_error = abs_error[:, :, i]
                
                if torch.sum(feat_mask) > 0:
                    per_feature_mae[i] += torch.sum(feat_error).item()
                    per_feature_count[i] += torch.sum(feat_mask).item()

    # Final Results
    avg_mae = total_mae / len(test_loader)
    avg_mse = total_mse / len(test_loader)
    
    print("\n" + "="*40)
    print(f" GLOBAL RESULTS (Normalized Data)")
    print("="*40)
    print(f"  MSE Global : {avg_mse:.4f}")
    print(f"  MAE Global : {avg_mae:.4f} (Mean error in standard deviation)")
    
    print("\n" + "-"*40)
    print(" PRECISION PER VITAL SIGN (Approx.)")
    print("-"*40)
    
    feature_names = list(VITAL_IDS.values())
    
    # Standard deviation estimates (for mental de-normalization)
    # These values are clinical approximations to give meaning to the numbers
    std_approx = {
        'HeartRate': 20.0,      # bpm
        'SystolicBP': 25.0,     # mmHg
        'DiastolicBP': 15.0,    # mmHg
        'SpO2': 3.0,            # %
        'RespRate': 5.0,        # rpm
        'Temperature': 1.5      # °F
    }
    
    for i in range(n_features):
        name = feature_names[i] if i < len(feature_names) else f"Feat {i}"
        
        # Normalized MAE (calculated by the model)
        norm_mae = per_feature_mae[i] / (per_feature_count[i] + 1e-8)
        
        # Estimated real MAE (by multiplying by the clinical standard deviation)
        std = std_approx.get(name, 1.0)
        real_mae = norm_mae * std
        
        unit = "bpm" if "Rate" in name else "mmHg" if "BP" in name else "%" if "SpO2" in name else "°F"
        
        print(f"  {name:15s} : Mean error {real_mae:.2f} {unit} (Norm: {norm_mae:.3f})")

    print("\nTest completed.")

if __name__ == "__main__":
    evaluate_model()