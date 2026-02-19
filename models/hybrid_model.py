import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .videomamba import VideoMamba
from .vjepa import VJEPAPredictor, SpatiotemporalBlockMasking, EMATargetEncoder
from .heads import build_head


class HybridVJEPAVideoMamba(nn.Module):
    """Complete Hybrid V-JEPA × VideoMamba model.

    Args:
        num_classes:    Number of action classes.
        num_frames:     Input temporal length. Default 16.
        img_size:       Spatial resolution. Default 224.
        embed_dim:      Backbone hidden dim. Default 384.
        depth:          Number of Mamba blocks. Default 24.
        d_state:        SSM state dim. Default 16.
        d_conv:         Conv kernel. Default 4.
        expand:         SSM expand. Default 2.
        drop_path_rate: Stochastic depth. Default 0.1.
        bidirectional:  Bidirectional scan. Default True.
        head_type:      Classification head. Default "attentive_probe".
        pred_dim:       Predictor dimension. Default 192.
        pred_depth:     Predictor Transformer depth. Default 4.
        mask_ratio:     V-JEPA masking ratio. Default 0.75.
        alpha:          Joint loss weight. Default 0.3.
        ema_momentum:   EMA initial momentum. Default 0.996.
    """

    def __init__(
        self,
        num_classes,
        num_frames=16,
        img_size=224,
        embed_dim=384,
        depth=24,
        d_state=16,
        d_conv=4,
        expand=2,
        drop_path_rate=0.1,
        bidirectional=True,
        head_type="attentive_probe",
        pred_dim=192,
        pred_depth=4,
        mask_ratio=0.75,
        alpha=0.3,
        ema_momentum=0.996,
    ):
        super().__init__()
        self.alpha = alpha
        self.mask_ratio = mask_ratio

        # 1. VideoMamba backbone
        self.backbone = VideoMamba(
            num_frames=num_frames,
            img_size=img_size,
            in_channels=3,  # RGB only
            embed_dim=embed_dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path_rate=drop_path_rate,
            bidirectional=bidirectional,
        )

        # 2. Classification head
        self.head = build_head(head_type, embed_dim, num_classes)

        # 3. V-JEPA components (only needed if α > 0)
        num_patches = self.backbone.num_patches
        self.masking = SpatiotemporalBlockMasking(
            mask_ratio=mask_ratio,
            num_temporal=self.backbone.num_temporal,
            num_spatial_h=self.backbone.num_spatial_h,
            num_spatial_w=self.backbone.num_spatial_w,
        )
        self.predictor = VJEPAPredictor(
            context_dim=embed_dim,
            pred_dim=pred_dim,
            depth=pred_depth,
            num_patches=num_patches,
        )

        # 4. EMA target encoder (created externally, but placeholder)
        self.ema_momentum = ema_momentum
        self._ema_encoder = None

        # Loss
        self.cls_criterion = nn.CrossEntropyLoss()

    def init_ema(self):
        """Initialize EMA target encoder. Call after model is on device."""
        self._ema_encoder = EMATargetEncoder(self.backbone, self.ema_momentum)
        return self._ema_encoder

    @property
    def ema_encoder(self):
        return self._ema_encoder

    def forward(self, x, labels=None, compute_jepa=True):
        """
        Args:
            x:            (B, C, T, H, W) video tensor.
            labels:       (B,) class labels. None for inference.
            compute_jepa: Whether to compute V-JEPA loss (set False for inference).
        Returns:
            dict with keys:
                'logits': (B, num_classes)
                'loss':   scalar (only if labels provided)
                'loss_cls': classification loss
                'loss_jepa': V-JEPA loss (0 if not computed)
        """
        # Backbone forward → full features
        features = self.backbone(x)               # (B, L+1, D)

        # Classification
        logits = self.head(features)              # (B, num_classes)

        result = {'logits': logits}

        if labels is not None:
            loss_cls = self.cls_criterion(logits, labels)
            result['loss_cls'] = loss_cls

            # V-JEPA loss
            loss_jepa = torch.tensor(0.0, device=x.device)
            if compute_jepa and self.alpha > 0 and self._ema_encoder is not None:
                loss_jepa = self._compute_jepa_loss(x, features)

            result['loss_jepa'] = loss_jepa
            result['loss'] = loss_cls + self.alpha * loss_jepa

        return result

    def _compute_jepa_loss(self, x, context_features):
        """Compute V-JEPA masked prediction loss.

        Loss = MSE(normalize(pred), normalize(target))  — Eq. (6)
        """
        B = x.shape[0]
        device = x.device

        # Generate masks
        mask, visible_idx, masked_idx = self.masking(B, device)

        # Get visible features from context encoder (exclude CLS)
        patch_features = context_features[:, 1:]  # (B, L, D)
        visible_features = torch.gather(
            patch_features, dim=1,
            index=visible_idx.unsqueeze(-1).expand(-1, -1, patch_features.shape[-1]).clamp(0, patch_features.shape[1] - 1)
        )

        # Predict masked features
        predictions = self.predictor(visible_features, visible_idx, masked_idx)

        # Get target features from EMA encoder
        with torch.no_grad():
            target_features_full = self._ema_encoder.forward(x)  # (B, L+1, D)
            target_patches = target_features_full[:, 1:]          # (B, L, D)
            targets = torch.gather(
                target_patches, dim=1,
                index=masked_idx.unsqueeze(-1).expand(-1, -1, target_patches.shape[-1]).clamp(0, target_patches.shape[1] - 1)
            )

        # Normalized MSE loss
        pred_norm = F.normalize(predictions, dim=-1)
        target_norm = F.normalize(targets, dim=-1)
        loss = F.mse_loss(pred_norm, target_norm)

        return loss

    def get_param_groups(self, weight_decay=0.05):
        """Separate parameters for weight decay.
        No decay on: biases, LayerNorm, positional embeddings.
        """
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or 'bias' in name or 'norm' in name or 'embed' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]

    def inference(self, x):
        """Inference-only forward pass. Predictor is not used → O(L)."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits
