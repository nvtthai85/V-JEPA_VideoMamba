import os
import time
import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from utils.metrics import AverageMeter, accuracy
from utils.logger import TrainingLogger
from utils.scheduler import CosineAnnealingWarmup
from utils.checkpoint import save_checkpoint
from models.vjepa import EMATargetEncoder


class Trainer:
    """Complete training pipeline.

    Args:
        model:        HybridVJEPAVideoMamba instance.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        config:       Dict with training hyperparameters.
        device:       torch.device.
    """

    def __init__(self, model, train_loader, val_loader, config, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.model.to(self.device)

        # Hyperparameters from config (matching paper Table 1)
        self.epochs = config.get('epochs', 100)
        self.lr = config.get('lr', 5e-4)
        self.weight_decay = config.get('weight_decay', 0.05)
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.min_lr = config.get('min_lr', 1e-6)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.patience = config.get('patience', 15)
        self.alpha = config.get('alpha', 0.3)
        self.ema_momentum_base = config.get('ema_momentum', 0.996)

        # Optimizer: AdamW (β₁=0.9, β₂=0.999)
        param_groups = self.model.get_param_groups(self.weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups, lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )

        # Scheduler: cosine + linear warmup
        self.scheduler = CosineAnnealingWarmup(
            self.optimizer, self.warmup_epochs, self.epochs, self.min_lr
        )

        # AMP
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)

        # EMA target encoder
        self.ema_encoder = None
        if self.alpha > 0:
            self.ema_encoder = self.model.init_ema()

        # Logger
        exp_name = config.get('exp_name', 'hybrid_vjepa')
        self.logger = TrainingLogger(
            log_dir=config.get('log_dir', 'logs'),
            exp_name=exp_name,
        )

        # State
        self.best_acc = 0.0
        self.epochs_no_improve = 0
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        self.checkpoint_name = config.get('checkpoint_name', f'{exp_name}_best.pth')

    def train(self):
        """Full training loop. Returns best validation accuracy."""
        self.logger.log(f"Training on {self.device}")
        self.logger.log(f"  Epochs: {self.epochs} | LR: {self.lr} | α: {self.alpha}")
        self.logger.log(f"  AMP: {self.use_amp} | Grad clip: {self.grad_clip}")
        self.logger.log(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        total_steps = self.epochs * len(self.train_loader)
        global_step = 0

        for epoch in range(1, self.epochs + 1):
            # Train one epoch
            train_metrics, global_step = self._train_epoch(epoch, global_step, total_steps)

            # Validate
            val_metrics = self._validate(epoch)

            # Scheduler step
            self.scheduler.step()

            # Log
            metrics = {
                'epoch': epoch,
                'lr': self.optimizer.param_groups[0]['lr'],
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
            }
            self.logger.log_epoch(metrics)

            # Early stopping / best checkpoint
            val_acc = val_metrics['acc1']
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.epochs_no_improve = 0
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_acc': self.best_acc,
                        'config': self.config,
                    },
                    os.path.join(self.checkpoint_dir, self.checkpoint_name),
                    is_best=True,
                )
            else:
                self.epochs_no_improve += 1

            self.logger.log(
                f"Epoch {epoch}: Val Acc={val_acc:.2f}% | "
                f"Best={self.best_acc:.2f}% | "
                f"No improve: {self.epochs_no_improve}/{self.patience}"
            )

            if self.epochs_no_improve >= self.patience:
                self.logger.log(f"Early stopping at epoch {epoch}")
                break

        self.logger.log(f"Best Val Acc: {self.best_acc:.2f}%")
        return self.best_acc

    def _train_epoch(self, epoch, global_step, total_steps):
        self.model.train()
        loss_m = AverageMeter('loss')
        cls_m = AverageMeter('cls')
        jepa_m = AverageMeter('jepa')
        acc_m = AverageMeter('acc')

        for batch_idx, (videos, labels) in enumerate(self.train_loader):
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            compute_jepa = self.alpha > 0

            with autocast(enabled=self.use_amp):
                result = self.model(videos, labels, compute_jepa=compute_jepa)
                loss = result['loss']

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update EMA encoder
            if self.ema_encoder is not None:
                momentum = EMATargetEncoder.cosine_momentum_schedule(
                    self.ema_momentum_base, global_step, total_steps
                )
                self.ema_encoder.update(self.model.backbone, momentum)

            # Metrics
            bs = videos.size(0)
            loss_m.update(loss.item(), bs)
            cls_m.update(result['loss_cls'].item(), bs)
            jepa_m.update(result.get('loss_jepa', torch.tensor(0)).item(), bs)
            acc1 = accuracy(result['logits'], labels, topk=(1,))[0]
            acc_m.update(acc1, bs)

            global_step += 1

        return {
            'loss': loss_m.avg,
            'loss_cls': cls_m.avg,
            'loss_jepa': jepa_m.avg,
            'acc1': acc_m.avg,
        }, global_step

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        loss_m = AverageMeter()
        acc_m = AverageMeter()

        for videos, labels in self.val_loader:
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model.inference(videos)
                loss = nn.functional.cross_entropy(logits, labels)

            acc1 = accuracy(logits, labels, topk=(1,))[0]
            bs = videos.size(0)
            loss_m.update(loss.item(), bs)
            acc_m.update(acc1, bs)

        return {'loss': loss_m.avg, 'acc1': acc_m.avg}
