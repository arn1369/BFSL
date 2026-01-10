import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from utils import MIMICPipeline, MIMICDataset, VITAL_IDS
from fsl import MIMICPredictor

def visualize_reconstruction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = MIMICPipeline()
    
    if os.path.exists("./saves/cache_mimic.pkl"):
        df = pd.read_pickle("./saves/cache_mimic.pkl")
    else:
        df = pipeline.load_cohort(n_patients=100)

    # 50% mask ratio
    dataset = MIMICDataset(df, window_size=24, mask_ratio=0.5) 
    
    # load model
    n_features = 6
    model = MIMICPredictor(n_assets=n_features, window_size=24).to(device)
    try:
        model.load_state_dict(torch.load("./saves/mimic_fsl.pth", map_location=device))
        print("Successfully loaded the trained model.")
    except Exception as e:
        print(f"Error while loading the model : {e}")
        return

    model.eval()

    # take an example
    # We look for an interesting example (not just a flat line)
    idx = np.random.randint(0, len(dataset))
    inputs, target_clean, mask = dataset[idx]
    
    # Prepare for the model
    inputs_dev = [x.unsqueeze(0).to(device) for x in inputs]
    
    with torch.no_grad():
        _, res = model(inputs_dev)
        reconstructed = torch.stack(res['outputs'], dim=2).squeeze(0).cpu().numpy() # (Time, Feats)
    
    target_clean = target_clean.numpy()
    mask = mask.numpy()
    
    # The masked input (what the model saw)
    input_visible = target_clean.copy()
    input_visible[mask == 1] = np.nan # We put NaN where it was hidden to not display it

    # 4. Plotting
    features = list(VITAL_IDS.values())
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 15), sharex=True)
    
    time_axis = np.arange(24)
    
    for i in range(n_features):
        ax = axes[i]
        feature_name = features[i] if i < len(features) else f"Feat {i}"
        
        # true data
        ax.plot(time_axis, target_clean[:, i], color='green', label='Ground Truth', linewidth=2, alpha=0.5)
        
        # reconstructed data
        ax.plot(time_axis, reconstructed[:, i], color='red', linestyle='--', label='FSL Reconstruction')
        
        # visible input
        # If there is a gap in the blue points, the model had to invent the red line!
        ax.scatter(time_axis, input_visible[:, i], color='blue', s=20, label='Visible Input', zorder=5)
        
        ax.set_ylabel(feature_name)
        if i == 0:
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"FSL Reconstruction (MIMIC-IV) - H1 Score: {res['h1_score'].item():.4f}")
    plt.xlabel("Hours")
    plt.tight_layout()
    plt.savefig("./visuals/reconstruction.png")

if __name__ == "__main__":
    visualize_reconstruction()