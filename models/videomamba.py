import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_block import MambaBlock


class TubeletEmbedding(nn.Module):
    """3D Patch (Tubelet) Embedding via Conv3D.

    Converts video (B, C, T, H, W) into a sequence of D-dim tokens.
    Tubelet = temporal_size × patch_size × patch_size.

    Args:
        in_channels:    Input channels (3 for RGB).
        embed_dim:      Token embedding dimension D.
        tubelet_size:   Temporal patch size (default 2).
        patch_size:     Spatial patch size (default 16).
    """

    def __init__(self, in_channels=3, embed_dim=384, tubelet_size=2, patch_size=16):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            tokens: (B, L, D) where L = (T/t) × (H/p) × (W/p)
        """
        x = self.proj(x)           # (B, D, T', H', W')
        B, D = x.shape[:2]
        x = x.flatten(2).transpose(1, 2)  # (B, L, D)
        return x


class VideoMamba(nn.Module):
    """VideoMamba backbone with bidirectional selective state space scanning.

    Args:
        num_frames:     Number of input frames T. Default 16.
        img_size:       Spatial resolution (H=W). Default 224.
        in_channels:    Input channels. Default 3 (RGB only).
        embed_dim:      Hidden dimension D. Default 384.
        depth:          Number of Mamba blocks. Default 24.
        d_state:        SSM state dimension N. Default 16.
        d_conv:         Conv kernel size. Default 4.
        expand:         SSM expansion factor. Default 2.
        drop_path_rate: Maximum stochastic depth rate. Default 0.1.
        tubelet_size:   Temporal patch size. Default 2.
        patch_size:     Spatial patch size. Default 16.
        bidirectional:  Use bidirectional scanning. Default True.
    """

    def __init__(
        self,
        num_frames=16,
        img_size=224,
        in_channels=3,
        embed_dim=384,
        depth=24,
        d_state=16,
        d_conv=4,
        expand=2,
        drop_path_rate=0.1,
        tubelet_size=2,
        patch_size=16,
        bidirectional=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.bidirectional = bidirectional
        self.num_frames = num_frames
        self.img_size = img_size

        # Tubelet embedding
        self.patch_embed = TubeletEmbedding(in_channels, embed_dim, tubelet_size, patch_size)

        # Token count
        self.num_temporal = num_frames // tubelet_size
        self.num_spatial_h = img_size // patch_size
        self.num_spatial_w = img_size // patch_size
        self.num_patches = self.num_temporal * self.num_spatial_h * self.num_spatial_w

        # [CLS] token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Stochastic depth: linearly increasing drop rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Forward-direction Mamba blocks
        self.blocks_fwd = nn.ModuleList([
            MambaBlock(embed_dim, d_state, d_conv, expand, drop_path=dpr[i])
            for i in range(depth)
        ])

        # Backward-direction blocks (separate weights for bidirectional)
        if bidirectional:
            self.blocks_bwd = nn.ModuleList([
                MambaBlock(embed_dim, d_state, d_conv, expand, drop_path=dpr[i])
                for i in range(depth)
            ])

        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) video tensor, C=3 (RGB).
        Returns:
            features: (B, L+1, D) including [CLS] at position 0.
        """
        B = x.shape[0]

        # Tubelet embedding → (B, L, D)
        tokens = self.patch_embed(x)

        # Prepend [CLS]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)     # (B, L+1, D)

        # Add positional embedding
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        # Forward scan
        fwd = tokens
        for blk in self.blocks_fwd:
            fwd = blk(fwd)

        if self.bidirectional:
            # Backward scan: flip patch tokens (keep CLS at position 0)
            bwd_patches = tokens[:, 1:].flip(1)        # reverse patch order
            bwd = torch.cat([tokens[:, :1], bwd_patches], dim=1)
            for blk in self.blocks_bwd:
                bwd = blk(bwd)
            # Flip back and combine
            bwd_patches = bwd[:, 1:].flip(1)
            bwd = torch.cat([bwd[:, :1], bwd_patches], dim=1)
            features = fwd + bwd                        # Eq. (4) in paper
        else:
            features = fwd

        features = self.norm(features)
        return features

    @property
    def num_tokens(self):
        return self.num_patches + 1  # patches + CLS

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
