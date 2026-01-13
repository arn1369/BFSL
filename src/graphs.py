import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from utils import MIMICPipeline, MIMICDataset, VITAL_IDS
from fsl import MIMICPredictor

# Graph configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def load_resources():
    """
    Loads data validation split and pre-trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = MIMICPipeline()
    
    if os.path.exists("./saves/cache_mimic.pkl"):
        df = pd.read_pickle("./saves/cache_mimic.pkl")
    else:
        raise Exception("Error: Cache not found! Train model first.")

    # Split dataset to retrieve the validation set (last 20%)
    all_ids = df.index.get_level_values(0).unique()
    val_ids = all_ids[int(len(all_ids)*0.8):]
    df_val = df.loc[val_ids]
    
    # Initialize dataset (using training stats ideally, re-calculated here for simplicity)
    dataset = MIMICDataset(df_val, window_size=24, mask_ratio=0.30)
    
    n_features = len(VITAL_IDS)
    model = MIMICPredictor(n_assets=n_features, window_size=24).to(device)
    
    try:
        model.load_state_dict(torch.load("./saves/mimic_fsl.pth", map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

    return model, dataset, device, n_features

def plot_adjacency(model, dataset, device):
    """
    Visualizes learned topological relationships across all hierarchy levels.
    Displays interactions from Fine (Organs) to Coarse (Latent clusters).
    """
    print("Generating adjacency matrices for all levels...")
    model.eval()
    
    # Fetch a single batch to compute average attention
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    inputs, _, _ = next(iter(loader))
    inputs = [x.to(device) for x in inputs]
    
    with torch.no_grad():
        _, res = model(inputs)
        
    # Retrieve adjacency pyramid: [Fine (6x6), Medium (3x3), Coarse (1x1)]
    adjs = res.get('adjacency_pyramid', [])
    if not adjs:
        print("No adjacency pyramid found in output.")
        return

    n_levels = len(adjs)
    fig, axes = plt.subplots(1, n_levels, figsize=(6 * n_levels, 6))
    if n_levels == 1: axes = [axes] # Handle single level case

    base_features = list(VITAL_IDS.values())

    for i, adj in enumerate(adjs):
        # Average over batch dimensions -> Shape: (N_features, N_features)
        mat = adj.mean(dim=0).cpu().numpy() 
        n_dim = mat.shape[0]
        
        # Dynamic labeling based on hierarchy level
        if i == 0:
            # Level 0: Physical Organs
            labels = base_features
            title = "Lvl 0: Fine (Physical)"
        else:
            # Level > 0: Latent Clusters
            labels = [f"Cluster {k}" for k in range(n_dim)]
            title = f"Lvl {i}: Coarse (Latent)"

        # Plot heatmap
        sns.heatmap(mat, ax=axes[i], cmap="viridis", annot=True, fmt=".2f", 
                    square=True, xticklabels=labels, yticklabels=labels, cbar=False)
        axes[i].set_title(title)

    plt.suptitle("Hierarchical Learned Interactions (Adjacency Pyramid)", fontsize=16)
    plt.tight_layout()
    plt.savefig("./visuals/adjacency_matrices.png")
    print("Saved: ./visuals/adjacency_matrices.png")

def plot_scatter_performance(model, dataset, device):
    """
    Compares Ground Truth vs. Prediction on masked data only.
    Diagnostic:
    - Diagonal (y=x) -> Good performance.
    - Horizontal line -> Model collapse (predicting mean).
    """
    print("Generating Scatter Plot...")
    model.eval()
    
    all_preds = []
    all_targets = []
    
    # Process a subset of batches to avoid overcrowding the plot
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    with torch.no_grad():
        for i, (inputs, targets, mask) in enumerate(loader):
            if i > 5: break # Stop after ~190 windows
            inputs = [x.to(device) for x in inputs]
            targets = targets.to(device)
            mask = mask.to(device)
            
            _, res = model(inputs)
            preds = torch.stack(res['outputs'], dim=2)
            
            # Filter: Evaluate only on masked (hidden) data points
            mask_bool = mask.bool()
            
            all_preds.append(preds[mask_bool].cpu().numpy())
            all_targets.append(targets[mask_bool].cpu().numpy())
            
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.1, s=10, color="purple")
    
    # Ideal prediction line (Identity)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal ($y=x$)")
    
    plt.xlabel("Ground Truth (Normalized)")
    plt.ylabel("FSL Prediction (Normalized)")
    plt.title("Reality vs. Prediction (Masked Data Only)")
    plt.legend()
    plt.grid(True)
    plt.savefig("./visuals/scatter_pred.png")
    print("Saved: ./visuals/scatter_pred.png")

def plot_enhanced_timeseries(model, dataset, device, n_features):
    """
    Visualizes time-series reconstruction with explicit masked regions.
    """
    print("Generating time-series visualization...")
    model.eval()
    
    # Select a random sample
    idx = np.random.randint(0, len(dataset))
    inputs, target_clean, mask = dataset[idx]
    
    # Heuristic: Search for a sample with high variance (non-flat signal)
    attempts = 0
    while torch.std(target_clean) < 0.5 and attempts < 10:
        idx = np.random.randint(0, len(dataset))
        inputs, target_clean, mask = dataset[idx]
        attempts += 1

    inputs_dev = [x.unsqueeze(0).to(device) for x in inputs]
    
    with torch.no_grad():
        _, res = model(inputs_dev)
        # Combine outputs: (Batch, Time, Feats) -> (Time, Feats)
        reconstructed = torch.stack(res['outputs'], dim=2).squeeze(0).cpu().numpy()
    
    target_clean = target_clean.numpy()
    mask = mask.numpy()
    
    features = list(VITAL_IDS.values())
    time = np.arange(24)
    
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 16), sharex=True)
    
    for i in range(n_features):
        ax = axes[i]
        feature_name = features[i]
        
        # 1. Highlight Masked Areas (Grey)
        masked_indices = np.where(mask[:, i] == 1)[0]
        for t in masked_indices:
            ax.axvspan(t-0.5, t+0.5, color='gray', alpha=0.2, lw=0)
            
        # 2. Plot Curves
        ax.plot(time, target_clean[:, i], 'g-', lw=2, label='Ground Truth', alpha=0.7)
        ax.plot(time, reconstructed[:, i], 'r--', lw=2, label='FSL Prediction')
        
        # 3. Plot Visible Inputs (Blue Dots)
        visible_mask = (mask[:, i] == 0)
        ax.scatter(time[visible_mask], target_clean[visible_mask, i], 
                   c='blue', s=30, zorder=5, label='Visible Input')

        ax.set_ylabel(feature_name)
        if i == 0:
            # Custom legend to include the grey area description
            from matplotlib.patches import Patch
            legend_elements = [
                plt.Line2D([0], [0], color='g', lw=2, label='Ground Truth'),
                plt.Line2D([0], [0], color='r', linestyle='--', lw=2, label='FSL Reconstruction'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label='Visible Input'),
                Patch(facecolor='gray', alpha=0.2, label='Masked Area (Imputed)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
    h1_score = res.get('h1_score', 0)
    # Check if h1_score is a tensor before formatting
    h1_val = h1_score.item() if isinstance(h1_score, torch.Tensor) else h1_score
    
    plt.suptitle(f"FSL Reconstruction (H1 Score: {h1_val:.4f})", fontsize=16)
    plt.xlabel("Time (Hours)")
    plt.tight_layout()
    plt.savefig("./visuals/reconstruction.png")
    print("Saved: ./visuals/reconstruction.png")

if __name__ == "__main__":
    os.makedirs("./visuals", exist_ok=True)
    
    model, dataset, device, n_feats = load_resources()
    if model:
        plot_adjacency(model, dataset, device)
        plot_scatter_performance(model, dataset, device)
        plot_enhanced_timeseries(model, dataset, device, n_feats)