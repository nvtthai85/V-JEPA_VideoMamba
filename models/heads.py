import torch
import torch.nn as nn
import torch.nn.functional as F


class CLSTokenHead(nn.Module):
    """Simple CLS token classification head.
    logits = Linear(LayerNorm(x[0]))
    """

    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, features):
        """features: (B, L+1, D) — CLS is at index 0."""
        cls = features[:, 0]              # (B, D)
        return self.fc(self.norm(cls))


class MeanPoolingHead(nn.Module):
    """Mean pooling over patch tokens (excludes CLS).
    logits = Linear(LayerNorm(mean(x[1:])))
    """

    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, features):
        """features: (B, L+1, D) — patches start at index 1."""
        pooled = features[:, 1:].mean(dim=1)  # (B, D)
        return self.fc(self.norm(pooled))


class AttentiveProbe(nn.Module):
    """Attentive Probe: cross-attention based classification head.

    Uses a learnable query to attend over all backbone features,
    then classifies through MLP.

    This head serves as the semantic anchor for classification.
    ~3.4M params, best results (+0.7% over CLS Token).
    """

    def __init__(self, embed_dim, num_classes, num_heads=6, mlp_dim=768):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(0.1),
        )

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, features):
        """features: (B, L+1, D)"""
        B = features.shape[0]
        q = self.query.expand(B, -1, -1)        # (B, 1, D)
        kv = features                            # (B, L+1, D)

        # Cross-attention: query attends to all features
        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, 1, D)
        q = self.norm1(q + attn_out)

        # MLP
        q = self.norm2(q + self.mlp(q))

        return self.head(q.squeeze(1))           # (B, num_classes)


def build_head(head_type, embed_dim, num_classes):
    """Factory function for classification heads.

    Args:
        head_type: One of "cls_token", "mean_pool", "attentive_probe".
        embed_dim: Backbone output dimension.
        num_classes: Number of action classes.
    """
    heads = {
        "cls_token": CLSTokenHead,
        "mean_pool": MeanPoolingHead,
        "attentive_probe": AttentiveProbe,
    }
    if head_type not in heads:
        raise ValueError(f"Unknown head: {head_type}. Choose from {list(heads.keys())}")
    return heads[head_type](embed_dim, num_classes)
