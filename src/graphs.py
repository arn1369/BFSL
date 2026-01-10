import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import MIMICPipeline, MIMICDataset, VITAL_IDS
from fsl import MIMICPredictor

def visualize_reconstruction():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = MIMICPipeline()
    
    # On charge le cache s'il existe (rapide)
    try:
        import os
        if os.path.exists("./saves/cache_mimic.pkl"):
            df = pd.read_pickle("./saves/cache_mimic.pkl")
        else:
            df = pipeline.load_cohort(n_patients=100)
    except:
        # Fallback si pas de cache, on recharge
        df = pipeline.load_cohort(n_patients=100)

    # On prend un mask_ratio élevé (50%) pour vraiment tester le modèle !
    dataset = MIMICDataset(df, window_size=24, mask_ratio=0.5) 
    
    # 2. Charger le modèle entraîné
    n_features = 6
    model = MIMICPredictor(n_assets=n_features, window_size=24).to(device)
    try:
        model.load_state_dict(torch.load("./saves/mimic_fsl.pth", map_location=device))
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Error while loading the model : {e}")
        return

    model.eval()

    # 3. Prendre un exemple
    # On cherche un exemple intéressant (pas juste une ligne plate)
    idx = np.random.randint(0, len(dataset))
    inputs, target_clean, mask = dataset[idx]
    
    # Préparer pour le modèle
    inputs_dev = [x.unsqueeze(0).to(device) for x in inputs] # Add batch dim
    
    with torch.no_grad():
        _, res = model(inputs_dev)
        reconstructed = torch.stack(res['outputs'], dim=2).squeeze(0).cpu().numpy() # (Time, Feats)
    
    target_clean = target_clean.numpy()
    mask = mask.numpy()
    
    # L'entrée masquée (ce que le modèle a vu)
    input_visible = target_clean.copy()
    input_visible[mask == 1] = np.nan # On met NaN là où c'était caché pour ne pas l'afficher

    # 4. Plotting
    features = list(VITAL_IDS.values())
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 15), sharex=True)
    
    time_axis = np.arange(24)
    
    for i in range(n_features):
        ax = axes[i]
        feature_name = features[i] if i < len(features) else f"Feat {i}"
        
        # Vraie donnée (Ligne verte continue)
        ax.plot(time_axis, target_clean[:, i], color='green', label='Vérité Terrain', linewidth=2, alpha=0.5)
        
        # Donnée reconstruite (Ligne rouge pointillée)
        ax.plot(time_axis, reconstructed[:, i], color='red', linestyle='--', label='Reconstruction FSL')
        
        # Ce que le modèle a vu (Points bleus)
        # S'il y a un trou dans les points bleus, le modèle a dû inventer la ligne rouge !
        ax.scatter(time_axis, input_visible[:, i], color='blue', s=20, label='Entrée Visible', zorder=5)
        
        ax.set_ylabel(feature_name)
        if i == 0:
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Reconstruction FSL (MIMIC-IV) - H1 Score: {res['h1_score'].item():.4f}")
    plt.xlabel("Heures")
    plt.tight_layout()
    plt.savefig("./visuals/reconstruction.png")

if __name__ == "__main__":
    visualize_reconstruction()