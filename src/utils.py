import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time

# Dictionnaire des IDs MIMIC-IV (Table chartevents)
# Ce sont les "Tickers" du corps humain
VITAL_IDS = {
    220045: 'HeartRate',
    220179: 'SystolicBP',
    220180: 'DiastolicBP',
    220277: 'SpO2',        # Saturation Oxygène
    220210: 'RespRate',    # Fréquence Respiratoire
    223761: 'Temperature'  # Fahrenheit
}

class MIMICPipeline:
    def __init__(self, root_dir="./docs/mimic-iv-3.1"):
        
        self.icu_path = os.path.join(root_dir, "icu")
        self.hosp_path = os.path.join(root_dir, "hosp")
        
    def load_cohort(self, n_patients=100):
        """
        Charge un sous-ensemble de patients pour éviter de saturer la RAM.
        """
        print(f"Chargement des données depuis {self.icu_path}...")
        
        if os.path.exists("./saves/cache_mimic.pkl"):
            print("Found already existing cache...")
            return pd.read_pickle("./saves/cache_mimic.pkl")
        
        # 1. Lire les admissions pour avoir les IDs patients
        # On prend juste les 100 premiers patients pour tester
        stays = pd.read_csv(os.path.join(self.icu_path, "icustays.csv"), nrows=n_patients)
        hadm_ids = stays['hadm_id'].unique()
        
        print(f"Extraction des signes vitaux pour {len(hadm_ids)} séjours...")
        
        start_time = time.time()
        
        # Read by chunk (too large file)
        chunksize = 10 ** 5
        data_fragments = []
        
        path_chart = os.path.join(self.icu_path, "chartevents.csv")
        
        # Lecture optimisée : on ne garde que les colonnes et items utiles
        with pd.read_csv(path_chart, chunksize=chunksize, usecols=['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum']) as reader:
            for chunk in reader:
                # Filtrer pour nos patients et nos variables
                filtered = chunk[
                    (chunk['hadm_id'].isin(hadm_ids)) & 
                    (chunk['itemid'].isin(VITAL_IDS.keys()))
                ]
                if not filtered.empty:
                    data_fragments.append(filtered)
                    
                # Sécurité pour ne pas tout lire pendant le dev
                if len(data_fragments) > 50:
                    print("Limite de fragments > 50, arrêt de la lecture.")
                    break
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nExtraction terminée en {duration:.2f} secondes.")
        
        if not data_fragments:
            raise ValueError("Aucune donnée trouvée. Vérifiez les chemins.")
            
        full_df = pd.concat(data_fragments)
        
        # 3. Pivot : Transformer en format Time-Series
        # Index: [Admission, Temps], Colonnes: [HR, BP, SpO2...]
        full_df['charttime'] = pd.to_datetime(full_df['charttime'])
        full_df = full_df.sort_values('charttime')
        
        # On renomme les IDs par des noms lisibles
        full_df['item_name'] = full_df['itemid'].map(VITAL_IDS)
        
        # On arrondit à l'heure près pour aligner les mesures
        full_df['time_bucket'] = full_df['charttime'].dt.round('h')
        
        pivot_df = full_df.pivot_table(
            index=['hadm_id', 'time_bucket'], 
            columns='item_name', 
            values='valuenum', 
            aggfunc='mean'
        )
        
        pivot_df.to_pickle("./saves/cache_mimic.pkl")
        print(f"Structured Data (saved to ./saves/cache_mimic.pkl) : {pivot_df.shape}")
        return pivot_df

class MIMICDataset(Dataset):
    def __init__(self, pivot_df, window_size=24, mask_ratio=0.2):
        """
        mask_ratio: Pourcentage de données qu'on va cacher artificiellement
        pour apprendre au modèle à les reconstruire (Imputation).
        """
        self.window_size = window_size
        self.mask_ratio = mask_ratio
        self.samples = []
        
        # Normalization
        scaler_mean = pivot_df.mean()
        scaler_std = pivot_df.std()
        normalized_df = (pivot_df - scaler_mean) / (scaler_std + 1e-6)
        
        # Forward fill for NaNs
        normalized_df = normalized_df.groupby('hadm_id').ffill() # forward fill
        normalized_df = normalized_df.groupby('hadm_id').bfill() # backward fill
        print(f"After forward-backward fill, there is {normalized_df.isna().sum().sum()} NaN values. Filled it with 0.")
        normalized_df = normalized_df.fillna(0.0)
        
        # Création des fenêtres glissantes
        for hadm_id in pivot_df.index.get_level_values(0).unique():
            patient_data = normalized_df.xs(hadm_id).values
            
            if len(patient_data) < window_size:
                continue
                
            for i in range(len(patient_data) - window_size):
                window = patient_data[i : i + window_size]
                self.samples.append(window)
                
        self.data = np.array(self.samples, dtype=np.float32)
        print(f"Dataset prêt : {self.data.shape} fenêtres de {window_size}h")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # x_clean est la vérité terrain
        x_clean = self.data[idx] 
        
        # On crée un masque artificiel (Self-Supervised Learning)
        # 1 = Donnée cachée, 0 = Donnée visible
        mask = np.random.binomial(1, self.mask_ratio, x_clean.shape).astype(np.float32)
        
        # x_masked est l'entrée corrompue que le modèle doit réparer
        # Si mask=1, on met la valeur à 0 (ou bruit)
        x_masked = x_clean * (1 - mask)
        
        # Tensorisation
        # Le modèle FSL attend une liste de tenseurs (un par asset/organe)
        # Shape actuelle : (Time, Feats). Transpose -> (Feats, Time)
        x_masked_T = x_masked.T
        inputs = [torch.tensor(x_masked_T[i]) for i in range(x_masked_T.shape[0])]
        
        return inputs, torch.tensor(x_clean), torch.tensor(mask)