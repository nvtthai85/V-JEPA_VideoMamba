"""Cosine annealing with linear warmup scheduler (paper Table 1)."""
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmup(_LRScheduler):
    """Cosine LR with linear warmup.

    Schedule: 0 → base_lr (linear warmup) → min_lr (cosine decay).

    Args:
        optimizer:      Torch optimizer.
        warmup_epochs:  Number of warmup epochs. Default 5.
        total_epochs:   Total training epochs. Default 100.
        min_lr:         Minimum learning rate. Default 1e-6.
    """

    def __init__(self, optimizer, warmup_epochs=5, total_epochs=100,
                 min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs
            ]
