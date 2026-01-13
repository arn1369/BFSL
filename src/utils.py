import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time

VITAL_IDS = {
    220045: 'HeartRate',
    220179: 'SystolicBP',
    220180: 'DiastolicBP',
    220277: 'SpO2',        # Oxygen Saturation
    220210: 'RespRate',    # Respiratory Rate
    223761: 'Temperature'  # Fahrenheit
}

class MIMICPipeline:
    def __init__(self, root_dir="./docs/mimic-iv-3.1"):
        
        self.icu_path = os.path.join(root_dir, "icu")
        self.hosp_path = os.path.join(root_dir, "hosp")
        
    def load_cohort(self, n_patients=100):
        """
        Load a subset of patients to avoid saturating RAM.
        """
        print(f"Loading data from {self.icu_path}...")
        
        if os.path.exists("./saves/cache_mimic.pkl"):
            print("Found already existing cache...")
            return pd.read_pickle("./saves/cache_mimic.pkl")
        
        # Read admissions to get patient IDs
        stays = pd.read_csv(os.path.join(self.icu_path, "icustays.csv"), nrows=n_patients)
        hadm_ids = stays['hadm_id'].unique()
        
        print(f"Extracting vital signs for {len(hadm_ids)} stays...")
        
        start_time = time.time()
        
        # Read by chunk (too large file)
        chunksize = 10 ** 5
        data_fragments = []
        
        path_chart = os.path.join(self.icu_path, "chartevents.csv")
        
        # Optimized reading: keep only useful columns and items
        with pd.read_csv(path_chart, chunksize=chunksize, usecols=['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum']) as reader:
            for chunk in reader:
                # Filter for our patients and variables
                filtered = chunk[
                    (chunk['hadm_id'].isin(hadm_ids)) & 
                    (chunk['itemid'].isin(VITAL_IDS.keys()))
                ]
                if not filtered.empty:
                    data_fragments.append(filtered)
                    
                # Avoid reading too much (crash RAM)
                if len(data_fragments) > 50:
                    print("Fragment limit > 50, stopping reading.")
                    break
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nExtraction completed in {duration:.2f} seconds.")
        
        if not data_fragments:
            raise ValueError("No data found. Check the paths.")
            
        full_df = pd.concat(data_fragments)
        
        # Pivot: Transform into Time-Series format
        # Index: [Admission, Time], Columns: [HR, BP, SpO2...]
        full_df['charttime'] = pd.to_datetime(full_df['charttime'])
        full_df = full_df.sort_values('charttime')
        
        # Rename IDs to readable names
        full_df['item_name'] = full_df['itemid'].map(VITAL_IDS)
        
        # Round to the nearest hour to align measurements
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
    def __init__(self, pivot_df, window_size=24, mask_ratio=0.2, stats=None):
        """
        mask_ratio: Percentage of data that will be artificially hidden
        to teach the model to reconstruct them (Imputation).
        """
        self.window_size = window_size
        self.mask_ratio = mask_ratio
        self.samples = []
        
        # Normalization
        if stats is None:
            self.scaler_mean = pivot_df.mean()
            self.scaler_std = pivot_df.std()
        else:
            self.scaler_mean, self.scaler_std = stats
        
        normalized_df = (pivot_df - self.scaler_mean) / (self.scaler_std + 1e-6)
        
        # Forward fill for NaNs
        normalized_df = normalized_df.groupby('hadm_id').ffill()
        print(f"After forward fill, there is {normalized_df.isna().sum().sum()} NaN values. Filled it with 0.")
        normalized_df = normalized_df.fillna(0.0)
        
        # Creation of sliding windows
        for hadm_id in pivot_df.index.get_level_values(0).unique():
            try: # avoid create sliding windows on two patients
                patient_data = normalized_df.xs(hadm_id).values
            except KeyError:
                continue
            
            if len(patient_data) < window_size:
                continue
                
            for i in range(len(patient_data) - window_size):
                window = patient_data[i : i + window_size]
                self.samples.append(window)
                
        self.data = np.array(self.samples, dtype=np.float32)
        print(f"Dataset ready: {self.data.shape} windows of {window_size}h")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # x_clean is the ground truth
        x_clean = self.data[idx] 
        
        # We create an artificial mask
        # 1 = Hidden data, 0 = Visible data
        mask = np.random.binomial(1, self.mask_ratio, x_clean.shape).astype(np.float32)
         
        # x_masked is the corrupted input that the model must repair
        # If mask=1, we set the value to 0 (or noise)
        x_masked = x_clean * (1 - mask)
        
        # Tensorization
        # The FSL model expects a list of tensors (one per asset/organ)
        # Current shape: (Time, Feats). Transpose -> (Feats, Time)
        x_masked_T = x_masked.T
        inputs = [torch.tensor(x_masked_T[i]) for i in range(x_masked_T.shape[0])]
        
        return inputs, torch.tensor(x_clean), torch.tensor(mask)
    
    def get_stats(self):
        """
        Return mean and std to normalize new data
        """
        return self.scaler_mean, self.scaler_std