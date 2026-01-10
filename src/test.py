import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import MIMICPipeline, MIMICDataset, VITAL_IDS
from fsl import MIMICPredictor

def evaluate_model():
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    WINDOW_SIZE = 24
    MASK_RATIO = 0.2 # On teste dans les mêmes conditions (20% de données manquantes)
    
    print("--- DÉMARRAGE DU TEST ---")
    
    # 2. Chargement de NOUVELLES données (Test Set)
    # Dans train.py, on a pris les 200 premiers patients.
    # Ici, on va essayer de charger une suite (ou on recharge tout et on split, 
    # mais pour l'exemple on va supposer qu'on recharge les mêmes pour valider l'apprentissage
    # ou idéalement on chargerait les patients 200 à 300).
    
    pipeline = MIMICPipeline()
    try:
        # On essaie de charger un "Test Set" (patients différents si possible)
        # Ici on recharge le cache pour simplifier, mais dans un vrai projet 
        # il faudrait scinder le dataset en deux au début.
        import os
        if os.path.exists("./saves/cache_mimic.pkl"):
            df = pd.read_pickle("./saves/cache_mimic.pkl")
            # On prend les 20% derniers pour le test (Hold-out set)
            split_idx = int(len(df) * 0.8)
            # Attention: il faut split par patient idéalement, mais ici on split par temps/ligne
            # C'est une approximation acceptable pour l'instant.
            df_test = df.iloc[split_idx:]
            print(f"Test Set chargé depuis le cache : {len(df_test)} lignes.")
        else:
            print("Cache non trouvé. Veuillez lancer train.py d'abord.")
            return
    except Exception as e:
        print(f"Erreur data : {e}")
        return

    test_ds = MIMICDataset(df_test, window_size=WINDOW_SIZE, mask_ratio=MASK_RATIO)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Charger le modèle
    n_features = 6 # HR, SysBP, DiaBP, SpO2, Resp, Temp
    model = MIMICPredictor(n_assets=n_features, window_size=WINDOW_SIZE).to(device)
    
    try:
        model.load_state_dict(torch.load("./saves/mimic_fsl.pth", map_location=device))
        print("Poids du modèle chargés.")
    except FileNotFoundError:
        print("Erreur : Fichier .pth introuvable.")
        return

    model.eval()
    
    # 4. Boucle d'évaluation
    total_mae = 0
    total_mse = 0
    
    # Pour stocker les erreurs par organe
    # Shape: (6,)
    per_feature_mae = np.zeros(n_features)
    per_feature_count = np.zeros(n_features)
    
    print(f"Évaluation sur {len(test_ds)} fenêtres...")
    
    with torch.no_grad():
        for inputs, targets, mask in test_loader:
            inputs = [x.to(device) for x in inputs]
            targets = targets.to(device)
            mask = mask.to(device)
            
            # Forward
            _, res = model(inputs)
            
            # Reconstruction
            reconstructed_stack = torch.stack(res['outputs'], dim=2) # (Batch, Time, Feats)
            
            # On ne calcule l'erreur QUE sur ce qui était masqué (le challenge)
            # Erreur Absolue (|Y_pred - Y_true|)
            abs_error = torch.abs(reconstructed_stack - targets) * mask
            
            # Mise à jour des métriques globales
            # On divise par le nombre d'éléments masqués pour avoir une vraie moyenne
            n_masked = torch.sum(mask) + 1e-8
            
            batch_mae = torch.sum(abs_error) / n_masked
            batch_mse = torch.sum((reconstructed_stack - targets)**2 * mask) / n_masked
            
            total_mae += batch_mae.item()
            total_mse += batch_mse.item()
            
            # Mise à jour des métriques par organe
            # On somme les erreurs pour chaque feature (dim=2) sur tout le batch et le temps
            # mask shape: (Batch, Time, Feats)
            for i in range(n_features):
                feat_mask = mask[:, :, i]
                feat_error = abs_error[:, :, i]
                
                if torch.sum(feat_mask) > 0:
                    per_feature_mae[i] += torch.sum(feat_error).item()
                    per_feature_count[i] += torch.sum(feat_mask).item()

    # 5. Résultats Finaux
    avg_mae = total_mae / len(test_loader)
    avg_mse = total_mse / len(test_loader)
    
    print("\n" + "="*40)
    print(f" RÉSULTATS GLOBAUX (Données Normalisées)")
    print("="*40)
    print(f"  MSE Global : {avg_mse:.4f}")
    print(f"  MAE Global : {avg_mae:.4f} (L'erreur moyenne en écart-type)")
    
    print("\n" + "-"*40)
    print(" PRÉCISION PAR SIGNE VITAL (Approx.)")
    print("-"*40)
    
    feature_names = list(VITAL_IDS.values())
    
    # Estimations des écarts-types standards (pour dé-normaliser mentalement)
    # Ces valeurs sont des approximations cliniques pour donner du sens aux chiffres
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
        
        # MAE normalisé (calculé par le modèle)
        norm_mae = per_feature_mae[i] / (per_feature_count[i] + 1e-8)
        
        # MAE réel estimé (en multipliant par l'écart-type clinique)
        std = std_approx.get(name, 1.0)
        real_mae = norm_mae * std
        
        unit = "bpm" if "Rate" in name else "mmHg" if "BP" in name else "%" if "SpO2" in name else "°F"
        
        print(f"  {name:15s} : Erreur moy. {real_mae:.2f} {unit} (Norm: {norm_mae:.3f})")

    print("="*40)
    print("Test terminé.")

if __name__ == "__main__":
    evaluate_model()