
"""
FSL Implementation
@author : Arnaud Ullens on 8th dec.2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np


class SoftOrthogonalRestriction(nn.Module):
    """
    Learns a linear mapping between dimensions while enforcing soft orthogonality.
    This helps preserve structural properties during dimension reduction/expansion.
    """
    def __init__(self, dim_source: int, dim_target: int):
        super().__init__()
        self.dim_source = dim_source
        self.dim_target = dim_target
        
        # SVD-based initialization for better convergence
        weight = torch.empty(dim_target, dim_source)
        nn.init.orthogonal_(weight)
        
        # Add noise to prevent collapse
        weight = weight + 0.01 * torch.randn_like(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def orthogonality_loss(self) -> torch.Tensor:
        """
        Calculates loss to maintain orthogonality (Gram matrix ≈ Identity).
        Includes spectral regularization to penalize small singular values.
        """
        if self.dim_target >= self.dim_source:
            gram = self.weight.t() @ self.weight
            eye = torch.eye(self.dim_source, device=self.weight.device)
        else:
            gram = self.weight @ self.weight.t()
            eye = torch.eye(self.dim_target, device=self.weight.device)
        
        # MSE between Gram matrix and Identity
        ortho_loss = F.mse_loss(gram, eye)
        
        # Spectral regularization: prevents the matrix from becoming rank-deficient
        s = torch.linalg.svdvals(self.weight)
        spectral_reg = torch.mean(torch.relu(0.1 - s))
        
        return ortho_loss + 0.1 * spectral_reg


class DynamicSheafLaplacian(nn.Module):
    """
    Core Sheaf Diffusion Layer.
    Learns the topology (adjacency) dynamically and diffuses information 
    across contexts (nodes) based on restriction maps.
    """
    def __init__(self, n_contexts: int, context_dims: List[int], 
                 attention_dim: int = 64, alpha: float = 0.5):
        super().__init__()
        self.n_contexts = n_contexts
        self.context_dims = context_dims
        self.attention_dim = attention_dim
        
        # Adaptive diffusion parameter (learned during training)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        
        # Projections for attention mechanism (Query/Key)
        self.query_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, attention_dim),
                nn.LayerNorm(attention_dim)
            ) for d in context_dims
        ])
        self.key_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, attention_dim),
                nn.LayerNorm(attention_dim)
            ) for d in context_dims
        ])
        
        # Pairwise restriction maps between all contexts
        self.restrictions = nn.ModuleDict()
        for i in range(n_contexts):
            for j in range(n_contexts):
                if i != j:
                    self.restrictions[f"{i}_{j}"] = SoftOrthogonalRestriction(
                        context_dims[i], context_dims[j]
                    )

    def compute_adjacency(self, sections: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the attention-based adjacency matrix between contexts.
        """
        batch_size = sections[0].shape[0]
        device = sections[0].device
        
        # Spatial pooling if input is 3D (to handle sequence data)
        pooled = []
        for s in sections:
            if s.dim() == 2:
                pooled.append(s)
            else:
                p = s.flatten(start_dim=1, end_dim=-2).mean(dim=1)
                pooled.append(p)

        # Compute Attention Scores
        Q = torch.stack([
            self.query_projs[i](pooled[i]) 
            for i in range(self.n_contexts)
        ], dim=1)
        
        K = torch.stack([
            self.key_projs[i](pooled[i]) 
            for i in range(self.n_contexts)
        ], dim=1)
        
        temperature = torch.exp(torch.tensor(np.log(self.attention_dim) / 2, device=device))
        scores = torch.bmm(Q, K.transpose(1, 2)) / temperature
        
        # Mask self-loops (diagonal)
        mask = torch.eye(self.n_contexts, device=device).bool().unsqueeze(0)
        scores = scores.masked_fill(mask, -1e9)
        adj = F.softmax(scores, dim=-1)
        
        return adj

    def forward(self, sections: List[torch.Tensor], diffusion_scale: float = 1.0) -> Tuple[List[torch.Tensor], torch.Tensor]:
        batch_size = sections[0].shape[0]
        adjacency = self.compute_adjacency(sections)
        
        # Calculate final alpha (diffusion strength)
        alpha = torch.sigmoid(self.alpha_logit) * diffusion_scale
        
        new_sections = []
        
        # Sheaf Diffusion Process: X_i = (1-a)X_i + a * sum(A_ij * Rho_ji(X_j)) 
        for i in range(self.n_contexts):
            current_s = sections[i]
            diffusion_term = torch.zeros_like(current_s)
            
            for j in range(self.n_contexts):
                if i == j:
                    continue
                
                # Transport data from j to i using restriction map
                transported = self.restrictions[f"{j}_{i}"](sections[j])
                weight = adjacency[:, i, j].view(batch_size, *([1] * (current_s.dim() - 1)))
                diffusion_term += weight * transported
            
            # Diffusion avec alpha adaptatif
            new_sections.append((1 - alpha) * current_s + alpha * diffusion_term)
        
        return new_sections, adjacency


class ScaleTransition(nn.Module):
    """
    Manages the data flow between hierarchy levels (Fine <-> Coarse).
    Uses Attention for pooling (down) and Projections for unpooling (up).
    """
    def __init__(self, n_contexts_fine: int, n_contexts_coarse: int, context_dim: int):
        super().__init__()
        self.n_fine = n_contexts_fine
        self.n_coarse = n_contexts_coarse
        self.context_dim = context_dim
        self.pool_ratio = n_contexts_fine // n_contexts_coarse
        
        assert n_contexts_fine % n_contexts_coarse == 0, "Fine must be divisible by coarse"
        
        # Components for attention-based pooling
        self.pool_query = nn.Linear(context_dim, context_dim)
        self.pool_key = nn.Linear(context_dim, context_dim)
        self.pool_value = nn.Linear(context_dim, context_dim)
        
        # Components for unpooling (upsampling)
        self.unpool_proj = nn.Sequential(
            nn.Linear(context_dim, context_dim * self.pool_ratio),
            nn.LayerNorm(context_dim * self.pool_ratio),
            nn.GELU(),
            nn.Linear(context_dim * self.pool_ratio, context_dim * self.pool_ratio)
        )
        
    def pool_up(self, sections_fine: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Aggregates fine-grained sections into coarse sections using attention.
        """
        sections_coarse = []
        
        for i in range(self.n_coarse):
            start_idx = i * self.pool_ratio
            end_idx = start_idx + self.pool_ratio
            
            # Stack relevant fine sections
            group = torch.stack(sections_fine[start_idx:end_idx], dim=1)
            group = group.flatten(start_dim=2)
            
            # Compute attention weights within the group
            Q = self.pool_query(group.mean(dim=1, keepdim=True))
            K = self.pool_key(group)
            V = self.pool_value(group)
            
            scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.context_dim)
            attn_weights = F.softmax(scores, dim=-1)
            
            # Weighted aggregation
            pooled = torch.bmm(attn_weights, V).squeeze(1)
            sections_coarse.append(pooled)
        
        return sections_coarse
    
    def unpool_down(self, sections_coarse: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Distributes coarse information back to fine sections via projection.
        """
        sections_fine = []
        
        for coarse_section in sections_coarse:
            # Expand dimensions
            unpooled = self.unpool_proj(coarse_section)  # (B, D * pool_ratio)
            unpooled = unpooled.view(-1, self.pool_ratio, self.context_dim)  # (B, pool_ratio, D)
            
            # Split into individual fine sections
            for k in range(self.pool_ratio):
                sections_fine.append(unpooled[:, k, :])
        
        return sections_fine


class HierarchicalFSL(nn.Module):
    """
    Main Hierarchical Folded Sheaf Laplacian Model.
    Structure acts like a Graph U-Net:
    1. Bottom-up: Encodes and coarsens data (extracts global features).
    2. Top-down: Refines features using residuals and diffusion.
    """
    def __init__(self, 
                 scales: List[int] = [16, 8, 4],
                 context_dim: int = 32,
                 attention_dim: int = 64,
                 diffusion_steps: List[int] = [2, 3, 4]):
        super().__init__()
        self.scales = scales
        self.context_dim = context_dim
        self.n_scales = len(scales)
        self.diffusion_steps = diffusion_steps
        
        assert len(diffusion_steps) == len(scales), "Need diffusion steps for each scale"
        
        # Encoders (applied only at the finest scale)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, context_dim),
                nn.LayerNorm(context_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(context_dim, context_dim),
                nn.LayerNorm(context_dim)
            )
            for _ in range(scales[0])
        ])
        
        # Diffusion layers for each scale
        self.sheaves = nn.ModuleList([
            DynamicSheafLaplacian(
                n_contexts=n,
                context_dims=[context_dim] * n,
                attention_dim=attention_dim,
                alpha=0.5  # Initial, sera appris
            )
            for n in scales
        ])
        
        # Transition layers (Pooling/Unpooling)
        self.transitions = nn.ModuleList([
            ScaleTransition(
                n_contexts_fine=scales[i],
                n_contexts_coarse=scales[i+1],
                context_dim=context_dim
            )
            for i in range(len(scales) - 1)
        ])
        
        # Decoders (applied only at the finest scale)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, context_dim)
            )
            for _ in range(scales[0])
        ])
    
    def forward(self, inputs: List[torch.Tensor]) -> Dict:
        """
        Forward pass with automatic volatility gating.
        inputs: List of tensors corresponding to the finest scale.
        """
        
        # Always diffuse information
        diffusion_scale = 1.0
        
        # Initial Encoding
        sections_fine = [enc(x) for enc, x in zip(self.encoders, inputs)]
        
        # --- BOTTOM-UP: Building the Pyramid ---
        pyramid = [sections_fine]
        adjacency_pyramid = []
        
        for i, (transition, sheaf) in enumerate(zip(self.transitions, self.sheaves[1:])):
            # Pool to coarser scale
            sections_coarse = transition.pool_up(pyramid[-1])
            
            # Diffuse information at this coarse level
            for _ in range(self.diffusion_steps[i+1]):
                sections_coarse, adj_coarse = sheaf(sections_coarse, diffusion_scale=diffusion_scale)
            
            pyramid.append(sections_coarse)
            adjacency_pyramid.append(adj_coarse)
        
        # --- TOP-DOWN: Refinement Cascade ---
        for i in reversed(range(len(self.transitions))):
            # Unpool from coarse to fine
            sections_from_coarse = self.transitions[i].unpool_down(pyramid[i+1])
            
            # Residual connection: Mix original fine features with upsampled coarse features
            residual_weight = 0.3
            for j in range(len(pyramid[i])):
                pyramid[i][j] = pyramid[i][j] + residual_weight * sections_from_coarse[j]
            
            # Re-diffuse at fine scale with enriched information
            for _ in range(self.diffusion_steps[i]):
                pyramid[i], adj_fine = self.sheaves[i](pyramid[i], diffusion_scale=diffusion_scale)
            
            # Update adjacency for return
            if i < len(adjacency_pyramid):
                adjacency_pyramid[i] = adj_fine
        
        # Final Reconstruction
        final_sections = pyramid[0]
        outputs = [dec(s) for dec, s in zip(self.decoders, final_sections)]
        
        # Compute Structural Losses
        h1_loss = self.compute_h1_multiscale(pyramid, adjacency_pyramid)
        ortho_loss = self.compute_ortho_loss()
        
        return {
            'outputs': outputs,
            'sections': final_sections,
            'pyramid': pyramid,
            'adjacency_pyramid': adjacency_pyramid,
            'h1_score': h1_loss,
            'ortho_loss': ortho_loss
        }
    
    def compute_h1_multiscale(self, pyramid: List[List[torch.Tensor]], 
                             adjacency_pyramid: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the H1 Cohomology across scales.
        Lower H1 = More global consistency/coherence.
        Higher weights are assigned to coarser scales (global structures).
        """
        total_h1 = 0.0
        scale_weights = [1.0, 1.5, 2.0]
        
        for scale_idx, (sections, adj) in enumerate(zip(pyramid[1:], adjacency_pyramid)):
            sheaf = self.sheaves[scale_idx + 1]
            n_contexts = len(sections)
            
            scale_h1 = 0.0
            count = 0
            
            for i in range(n_contexts):
                for j in range(n_contexts):
                    if i == j:
                        continue
                    
                    # Calculate consistency: Rho_ji(Rho_ij(x)) vs x
                    rho_ij = sheaf.restrictions[f"{i}_{j}"]
                    rho_ji = sheaf.restrictions[f"{j}_{i}"]
                    
                    cycle = rho_ji(rho_ij(sections[i]))
                    weight = adj[:, i, j].mean()
                    
                    # Only consider strong connections
                    if weight > 0.05:
                        diff = (cycle - sections[i]).pow(2).mean()
                        scale_h1 += weight * diff
                        count += 1
            
            if count > 0:
                scale_weight = scale_weights[min(scale_idx, len(scale_weights)-1)]
                total_h1 += scale_weight * (scale_h1 / count)
        
        return total_h1 / sum(scale_weights[:len(adjacency_pyramid)])
    
    def compute_ortho_loss(self) -> torch.Tensor:
        """
        Aggregates orthogonality loss from all restriction maps.
        """
        loss = 0.0
        count = 0
        
        for sheaf in self.sheaves:
            for restriction in sheaf.restrictions.values():
                loss += restriction.orthogonality_loss()
                count += 1
        
        return loss / count

class TopologicalContrastiveLoss(nn.Module):
    """
    Simple contrastive loss on H1 scores.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, h1_coherent: torch.Tensor, h1_incoherent: torch.Tensor) -> torch.Tensor:
       # Coherent data should have low H1, Incoherent data should have high H1
        loss_coherent = h1_coherent
        loss_incoherent = F.relu(self.margin - h1_incoherent)
        
        return loss_coherent + loss_incoherent
    
class TopologicalTripletLoss(nn.Module):
    """
    Advanced Triplet Loss optimized for Sheaf Cohomology (H1).
    Objectives:
    1. Structural Consistency: Real data should have low H1.
    2. Temporal Stability: Anchor and Positive should have similar H1.
    3. Discrimination: Anchor and Negative (noise) should have very different H1.
    """
    def __init__(self, margin: float = 0.5, structural_weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.structural_weight = structural_weight
    
    def forward(self, 
                h1_anchor: torch.Tensor, 
                h1_positive: torch.Tensor, 
                h1_negative: torch.Tensor,
                anchor_embedding: torch.Tensor = None,
                positive_embedding: torch.Tensor = None) -> torch.Tensor:
        
        # Structural Term: Minimize H1 for valid data (Anchor & Positive)
        structural_loss = h1_anchor + h1_positive
        
        # Triplet Term: Ensure Anchor is topologically closer to Positive than Negative
        d_pos = torch.abs(h1_anchor - h1_positive)
        d_neg = torch.abs(h1_anchor - h1_negative)
        
        triplet_loss = torch.relu(d_pos - d_neg + self.margin)
        
        # Standard embedding distance regularization
        embedding_loss = 0.0
        if anchor_embedding is not None and positive_embedding is not None:
            embedding_loss = F.mse_loss(anchor_embedding, positive_embedding)
            
        return self.structural_weight * structural_loss + triplet_loss + 0.1 * embedding_loss

class MIMICPredictor(nn.Module):
    def __init__(self, n_assets, window_size):
        super().__init__()
        
        #ANCHOR: Simple hierarchy, will probably change in the future
        self.fsl = HierarchicalFSL(
            scales=[n_assets, n_assets//2, 1], 
            context_dim=window_size,
            attention_dim=32,
            diffusion_steps=[1, 2, 2]
        )
        
        # Optional in BFSL but it's for compatibility with FSL structure
        self.head = nn.Sequential(
            nn.Linear(window_size, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_list):
        res = self.fsl(x_list)
        return getattr(res, 'outputs', res['outputs']), res