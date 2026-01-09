import numpy as np
import torch
from torch import nn
from collections import deque
from torch.utils.data import Dataset

from fsl import HierarchicalFSL

class MultiRegimeHMM:
    def __init__(self, inertia=0.95, init_mean=None, init_var=None):
        self.n_states = 4
        
        # TRANSITIONS "COLLANTES" (Sticky)
        self.trans_prob = np.array([
            [0.94, 0.04, 0.01, 0.01],  # Crise
            [0.05, 0.85, 0.08, 0.02],  # Volatil
            [0.01, 0.05, 0.85, 0.09],  # Normal
            [0.01, 0.04, 0.10, 0.85]   # Bull
        ])
        self.trans_prob = self.trans_prob / self.trans_prob.sum(axis=1, keepdims=True)
        
        self.state_prob = np.array([0.05, 0.15, 0.60, 0.20])
        self.smooth_prob = np.array([0.05, 0.15, 0.60, 0.20])
        
        # Baseline ancrée (Z-SCORES)
        self.means = np.array([3.0, 1.5, 0.0, -0.5]) 
        self.vars = np.array([1.0, 0.8, 0.5, 0.5])
        
        # EMA pour lisser l'observation
        self.ema_signal = None
        self.ema_alpha = 0.2

    def gaussian_pdf(self, x, mean, var):
        denom = np.sqrt(2 * np.pi * var) + 1e-8
        num = np.exp(-0.5 * ((x - mean)**2) / (var + 1e-8))
        return num / denom

    def update(self, observation):
        if self.ema_signal is None:
            self.ema_signal = observation
        else:
            self.ema_signal = self.ema_alpha * observation + (1 - self.ema_alpha) * self.ema_signal
        
        obs = self.ema_signal
        
        pred_prob = self.state_prob @ self.trans_prob
        
        likelihoods = np.array([
            self.gaussian_pdf(obs, self.means[i], self.vars[i])
            for i in range(self.n_states)
        ])
        
        if np.sum(likelihoods) < 1e-20:
            likelihoods = np.ones(self.n_states)
        
        unnorm_prob = likelihoods * pred_prob
        self.state_prob = unnorm_prob / (np.sum(unnorm_prob) + 1e-16)
        
        return self.state_prob

class RollingZScore:
    def __init__(self, window=60):
        self.window = window
        self.history = deque(maxlen=window)
    
    def update(self, value):
        self.history.append(value)
        if len(self.history) < 10:
            return 0.0 
        
        mean = np.mean(self.history)
        std = np.std(self.history) + 1e-6
        z_score = np.clip((value - mean) / std, -4.0, 4.0)
        return z_score

class RollingWindowDataset(Dataset):
    def __init__(self, dataframe, window_size=20):
        self.data = dataframe.values.astype(np.float32)
        self.window_size = window_size
    def __len__(self):
        return len(self.data) - self.window_size
    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size]
        target = self.data[idx + self.window_size]
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True) + 1e-6
        x_norm = (window - mean) / std
        x_norm_T = x_norm.T 
        inputs = [torch.tensor(x_norm_T[i]) for i in range(x_norm_T.shape[0])]
        return inputs, torch.tensor(target)

class FSLPredictor(nn.Module):
    """
    Wraps the Hierarchical FSL backbone with a prediction head.
    The FSL extracts features, and the head predicts the next return.
    """
    def __init__(self, n_assets, window_size):
        super().__init__()
        self.fsl = HierarchicalFSL(
            scales=[16, 4, 1], # Hierarchical reduction
            context_dim=window_size,
            attention_dim=32,
            diffusion_steps=[1, 2, 2]
        )
        # Simple linear head for forecasting
        self.head = nn.Sequential(
            nn.Linear(window_size, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_list):
        res = self.fsl(x_list)
        clean_sections = res['sections']
        
        # Predict separately for each asset using the shared head
        preds = []
        for section in clean_sections:
            preds.append(self.head(section))
            
        return torch.cat(preds, dim=1), res