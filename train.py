#!/usr/bin/env python3
"""
train.py — Main training entry point for Hybrid V-JEPA × VideoMamba.

Usage:
    # Full training with config
    python train.py --config configs/experiments/ntu60_cs_joint.yaml

    # Quick test with synthetic data (no GPU needed)
    python train.py --config configs/experiments/ntu60_cs_joint.yaml --dry_run --epochs 2

    # Override hyperparameters via CLI
    python train.py --config configs/experiments/ntu60_cs_joint.yaml --alpha 0.5 --lr 1e-4 --seed 43

    # Supervised only (no V-JEPA)
    python train.py --config configs/experiments/ntu60_cs_joint.yaml --alpha 0.0
"""

import os
import sys
import argparse
import random
import yaml
import numpy as np
import torch

from configs import get_default_config
from models import HybridVJEPAVideoMamba
from datasets.video_dataset import build_dataloaders
from engine.trainer import Trainer


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(args):
    """Merge: default config → YAML file → CLI overrides."""
    config = get_default_config()

    # Load YAML config
    if args.config and os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config:
            config.update(yaml_config)

    # CLI overrides
    cli_overrides = {
        'seed': args.seed,
        'epochs': args.epochs,
        'lr': args.lr,
        'alpha': args.alpha,
        'batch_size': args.batch_size,
        'head_type': args.head_type,
        'bidirectional': args.bidirectional,
        'mask_ratio': args.mask_ratio,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'dry_run': args.dry_run,
        'num_workers': args.num_workers,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            config[k] = v

    # Device
    if args.device == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config['device'] = args.device

    return config


def build_model(config):
    """Create model from config."""
    model = HybridVJEPAVideoMamba(
        num_classes=config['num_classes'],
        num_frames=config.get('num_frames', 16),
        img_size=config.get('img_size', 224),
        embed_dim=config.get('embed_dim', 384),
        depth=config.get('depth', 24),
        d_state=config.get('d_state', 16),
        d_conv=config.get('d_conv', 4),
        expand=config.get('expand', 2),
        drop_path_rate=config.get('drop_path_rate', 0.1),
        bidirectional=config.get('bidirectional', True),
        head_type=config.get('head_type', 'attentive_probe'),
        pred_dim=config.get('pred_dim', 192),
        pred_depth=config.get('pred_depth', 4),
        mask_ratio=config.get('mask_ratio', 0.75),
        alpha=config.get('alpha', 0.3),
        ema_momentum=config.get('ema_momentum', 0.996),
    )
    return model


def main():
    parser = argparse.ArgumentParser(description='Hybrid V-JEPA × VideoMamba Training')
    parser.add_argument('--config', type=str, required=True, help='YAML config path')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--head_type', type=str, default=None,
                        choices=['cls_token', 'mean_pool', 'attentive_probe'])
    parser.add_argument('--bidirectional', type=bool, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--dry_run', action='store_true', help='Use synthetic data')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    config = load_config(args)
    seed = config.get('seed', 42)
    set_seed(seed)

    device = torch.device(config['device'])

    # Print config
    print("\n" + "=" * 60)
    print("  Hybrid V-JEPA × VideoMamba — Training")
    print("=" * 60)
    print(f"  Config:       {args.config}")
    print(f"  Experiment:   {config.get('exp_name', 'unknown')}")
    print(f"  Dataset:      {config.get('dataset_name', 'unknown')}")
    print(f"  Classes:      {config['num_classes']}")
    print(f"  Alpha:        {config.get('alpha', 0.3)}")
    print(f"  Head:         {config.get('head_type', 'attentive_probe')}")
    print(f"  Bidirectional:{config.get('bidirectional', True)}")
    print(f"  Mask ratio:   {config.get('mask_ratio', 0.75)}")
    print(f"  Epochs:       {config.get('epochs', 100)}")
    print(f"  LR:           {config.get('lr', 5e-4)}")
    print(f"  Batch:        {config.get('batch_size', 16)}")
    print(f"  Seed:         {seed}")
    print(f"  Device:       {device}")
    print(f"  Dry run:      {config.get('dry_run', False)}")
    print("=" * 60 + "\n")

    # Build model
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Build data
    train_loader, val_loader = build_dataloaders(config)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, device)
    best_acc = trainer.train()

    print(f"\nFinal Val Acc: {best_acc:.2f}%")
    print(f"Best Val Acc: {best_acc:.2f}%")

    return best_acc


if __name__ == '__main__':
    main()
