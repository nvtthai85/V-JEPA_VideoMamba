import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatiotemporalBlockMasking(nn.Module):
    """Block-wise spatiotemporal masking for V-JEPA.

    Masks contiguous 3D blocks of tokens (not random per-token).
    More challenging than random masking — forces holistic understanding.

    Args:
        mask_ratio:    Fraction of tokens to mask. Default 0.75.
        num_temporal:  Temporal grid size (T/tubelet_size).
        num_spatial_h: Spatial height grid (H/patch_size).
        num_spatial_w: Spatial width grid (W/patch_size).
    """

    def __init__(self, mask_ratio=0.75, num_temporal=8, num_spatial_h=14, num_spatial_w=14):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.nT = num_temporal
        self.nH = num_spatial_h
        self.nW = num_spatial_w
        self.num_patches = self.nT * self.nH * self.nW

    def forward(self, batch_size, device):
        """Generate block mask.

        Returns:
            mask: (B, L) bool tensor. True = MASKED (to predict).
            visible_idx: (B, n_visible) indices of visible tokens.
            masked_idx:  (B, n_masked) indices of masked tokens.
        """
        num_masked = int(self.num_patches * self.mask_ratio)
        num_visible = self.num_patches - num_masked

        masks = []
        visible_indices = []
        masked_indices = []

        for _ in range(batch_size):
            mask = self._generate_block_mask(device)
            masks.append(mask)

            vis = mask.eq(False).nonzero(as_tuple=False).squeeze(-1)
            msk = mask.eq(True).nonzero(as_tuple=False).squeeze(-1)

            # Pad/truncate to fixed size
            vis = self._pad_or_truncate(vis, num_visible, device)
            msk = self._pad_or_truncate(msk, num_masked, device)

            visible_indices.append(vis)
            masked_indices.append(msk)

        mask = torch.stack(masks)                     # (B, L)
        visible_idx = torch.stack(visible_indices)    # (B, n_vis)
        masked_idx = torch.stack(masked_indices)      # (B, n_msk)

        return mask, visible_idx, masked_idx

    def _generate_block_mask(self, device):
        """Generate one block mask by placing random rectangular blocks."""
        mask = torch.zeros(self.nT, self.nH, self.nW, dtype=torch.bool, device=device)
        target = int(self.num_patches * self.mask_ratio)
        masked_count = 0
        max_attempts = 50

        for _ in range(max_attempts):
            if masked_count >= target:
                break
            # Random block dimensions (at least 1)
            bt = max(1, torch.randint(1, max(2, self.nT // 2), (1,)).item())
            bh = max(1, torch.randint(2, max(3, self.nH // 3), (1,)).item())
            bw = max(1, torch.randint(2, max(3, self.nW // 3), (1,)).item())
            # Random position
            st = torch.randint(0, max(1, self.nT - bt + 1), (1,)).item()
            sh = torch.randint(0, max(1, self.nH - bh + 1), (1,)).item()
            sw = torch.randint(0, max(1, self.nW - bw + 1), (1,)).item()
            mask[st:st+bt, sh:sh+bh, sw:sw+bw] = True
            masked_count = mask.sum().item()

        return mask.flatten()  # (L,)

    @staticmethod
    def _pad_or_truncate(idx, target_len, device):
        if len(idx) >= target_len:
            perm = torch.randperm(len(idx), device=device)[:target_len]
            return idx[perm]
        pad = torch.zeros(target_len - len(idx), dtype=idx.dtype, device=device)
        return torch.cat([idx, pad])


class TransformerBlock(nn.Module):
    """Standard Transformer block for the predictor (O(n²) attention)."""

    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class VJEPAPredictor(nn.Module):
    """V-JEPA Predictor: predicts masked token features from visible context.

    Lightweight Transformer that takes visible features + learnable mask tokens
    and outputs predictions for the masked positions.

    Complexity: O(n²) due to self-attention, where n = total tokens.

    Args:
        context_dim:  Backbone output dimension (e.g. 384).
        pred_dim:     Predictor hidden dimension (e.g. 192).
        depth:        Number of Transformer blocks. Default 4.
        num_heads:    Attention heads. Default 6.
        num_patches:  Total patch count (excluding CLS). Default 1568.
    """

    def __init__(self, context_dim=384, pred_dim=192, depth=4, num_heads=6,
                 num_patches=1568):
        super().__init__()
        self.context_dim = context_dim
        self.pred_dim = pred_dim
        self.num_patches = num_patches

        # Project from backbone dim to predictor dim
        self.input_proj = nn.Linear(context_dim, pred_dim)

        # Learnable mask tokens (one per masked position)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embedding for all positions
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, pred_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(pred_dim, num_heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(pred_dim)
        self.output_proj = nn.Linear(pred_dim, context_dim)

    def forward(self, visible_features, visible_idx, masked_idx):
        """
        Args:
            visible_features: (B, n_vis, context_dim) — backbone features of visible tokens.
            visible_idx:      (B, n_vis) — indices of visible tokens in the sequence.
            masked_idx:       (B, n_msk) — indices of masked tokens.
        Returns:
            predictions: (B, n_msk, context_dim) — predicted features for masked positions.
        """
        B, n_vis, _ = visible_features.shape
        n_msk = masked_idx.shape[1]

        # Project visible features to predictor dim
        vis = self.input_proj(visible_features)            # (B, n_vis, pred_dim)

        # Add positional embeddings to visible tokens
        vis_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1),
            dim=1,
            index=visible_idx.unsqueeze(-1).expand(-1, -1, self.pred_dim).clamp(0, self.num_patches - 1)
        )
        vis = vis + vis_pos

        # Create mask tokens with positional embedding
        msk = self.mask_token.expand(B, n_msk, -1)
        msk_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1),
            dim=1,
            index=masked_idx.unsqueeze(-1).expand(-1, -1, self.pred_dim).clamp(0, self.num_patches - 1)
        )
        msk = msk + msk_pos

        # Concatenate: [visible, mask_tokens]
        x = torch.cat([vis, msk], dim=1)               # (B, n_vis+n_msk, pred_dim)

        # Transformer blocks (O(n²) self-attention)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Extract only the masked position predictions
        predictions = x[:, n_vis:]                       # (B, n_msk, pred_dim)
        predictions = self.output_proj(predictions)      # (B, n_msk, context_dim)

        return predictions


class EMATargetEncoder:
    """Exponential Moving Average target encoder (no gradients).

    θ_target ← m · θ_target + (1 − m) · θ_context

    Momentum m follows cosine schedule: 0.996 → 1.0 over training.
    """

    def __init__(self, backbone, initial_momentum=0.996):
        self.target = copy.deepcopy(backbone)
        for p in self.target.parameters():
            p.requires_grad = False
        self.m = initial_momentum

    @torch.no_grad()
    def update(self, backbone, current_momentum=None):
        """Update target parameters via EMA."""
        m = current_momentum if current_momentum is not None else self.m
        for p_target, p_context in zip(self.target.parameters(), backbone.parameters()):
            p_target.data.mul_(m).add_(p_context.data, alpha=1.0 - m)

    @torch.no_grad()
    def forward(self, x):
        """Get target features (no gradient)."""
        self.target.eval()
        return self.target(x)

    def state_dict(self):
        return self.target.state_dict()

    def load_state_dict(self, sd):
        self.target.load_state_dict(sd)

    @staticmethod
    def cosine_momentum_schedule(base_m, step, total_steps):
        """Cosine schedule from base_m → 1.0."""
        return 1.0 - (1.0 - base_m) * (1.0 + math.cos(math.pi * step / total_steps)) / 2.0
