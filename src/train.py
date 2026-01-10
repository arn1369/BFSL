import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from utils import MIMICPipeline, MIMICDataset
from fsl import MIMICPredictor

def sharp_loss(pred, target, mask):
    """
    Combine L1 (Valeur) et Diff (Pente) pour éviter les lignes plates.
    """
    # A. L1 Loss classique (Valeur Absolue)
    # Punit l'erreur sans être trop sensible aux outliers (contrairement au carré)
    l1 = torch.abs(pred - target) * mask
    term_val = torch.sum(l1) / (torch.sum(mask) + 1e-8)
    
    # B. Derivative Loss (Pénalité de Forme)
    # On force la pente (t+1 - t) prédite à ressembler à la vraie pente
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]       # Pente prédite
    target_diff = target[:, 1:, :] - target[:, :-1, :] # Vraie pente
    
    # On adapte le masque car on a perdu 1 point de temps avec la diff
    mask_diff = mask[:, 1:, :] * mask[:, :-1, :] 
    
    l1_diff = torch.abs(pred_diff - target_diff) * mask_diff
    term_shape = torch.sum(l1_diff) / (torch.sum(mask_diff) + 1e-8)
    
    # On combine : 1.0 * Valeur + 1.0 * Forme
    return term_val + 1.0 * term_shape

def train_mimic_reconstruction():
    # --- CHARGEMENT DONNÉES ---
    pipeline = MIMICPipeline()
    try:
        # On charge les données (cache ou nouveau)
        import os
        if os.path.exists("./saves/cache_mimic.pkl"):
            import pandas as pd
            df = pd.read_pickle("./saves/cache_mimic.pkl")
            print("Cache chargé.")
        else:
            df = pipeline.load_cohort(n_patients=200)
    except Exception as e:
        print(f"Erreur data : {e}")
        return

    # Dataset
    dataset = MIMICDataset(df, window_size=24, mask_ratio=0.20)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Configuration
    n_features = df.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Initialisation du modèle pour {n_features} signes vitaux...")
    model = MIMICPredictor(n_assets=n_features, window_size=24).to(device)
    
    # --- CONFIGURATION ENTRAÎNEMENT ---
    EPOCHS = 30 # <--- AUGMENTÉ (pour laisser le temps au scheduler)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # NOUVEAU : Le Scheduler
    # "Si la loss ne baisse pas pendant 3 époques, divise le LR par 2"
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print("Début de l'entraînement (L1 + Shape Loss + Scheduler)...")
    
    for epoch in range(EPOCHS):
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
            
            # --- CALCUL DE LA LOSS ---
            reconstructed_stack = torch.stack(res['outputs'], dim=2)
            
            # 1. On utilise notre nouvelle fonction sharp_loss
            reconstruction_loss = sharp_loss(reconstructed_stack, targets, mask)
            
            # 2. On récupère le H1
            h1_loss = res['h1_score']
            
            # 3. Total (On pondère le H1)
            loss = reconstruction_loss + 0.1 * h1_loss
            
            loss.backward()
            
            # NOUVEAU : Clipping de Gradient
            # Empêche le modèle de faire des bonds trop grands si la Loss explose
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Logs
            total_loss += reconstruction_loss.item()
            total_h1 += h1_loss.item() if isinstance(h1_loss, torch.Tensor) else h1_loss
            
        # Moyennes de l'époque
        avg_loss = total_loss / len(loader)
        avg_h1 = total_h1 / len(loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss (Sharp): {avg_loss:.4f} | H1: {avg_h1:.6f}")
        
        scheduler.step(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}") # On l'affiche nous-mêmes
    
    # Sauvegarde
    torch.save(model.state_dict(), "./saves/mimic_fsl.pth")
    print("Modèle sauvegardé.")

if __name__ == "__main__":
    train_mimic_reconstruction()