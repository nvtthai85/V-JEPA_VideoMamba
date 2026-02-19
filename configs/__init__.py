"""Default configuration matching paper Table 1 hyperparameters."""


def get_default_config():
    """Return default config dict. YAML files and CLI args override these."""
    return {
        # Experiment
        'exp_name': 'hybrid_vjepa_videomamba',
        'dataset_name': 'NTU-60 CS',
        'seed': 42,

        # Architecture — VideoMamba-Small
        'embed_dim': 384,
        'depth': 24,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'drop_path_rate': 0.1,
        'bidirectional': True,
        'head_type': 'attentive_probe',

        # Input — RGB only
        'num_frames': 16,
        'img_size': 224,
        'in_channels': 3,
        'input_modality': 'RGB',
        'num_classes': 60,

        # V-JEPA
        'pred_dim': 192,
        'pred_depth': 4,
        'mask_ratio': 0.75,
        'ema_momentum': 0.996,
        'alpha': 0.3,

        # Optimizer — AdamW
        'lr': 5e-4,
        'weight_decay': 0.05,
        'betas': [0.9, 0.999],
        'eps': 1e-8,

        # Schedule — Cosine + warmup
        'epochs': 100,
        'warmup_epochs': 5,
        'min_lr': 1e-6,

        # Training
        'batch_size': 16,
        'grad_clip': 1.0,
        'use_amp': True,
        'patience': 15,

        # Augmentation (paper Table 1)
        'crop_scale': [0.5, 1.0],
        'hflip_prob': 0.5,
        'color_jitter': 0.4,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],

        # Paths
        'data_root': './data/ntu60',
        'train_annotation': './data/ntu60/annotations/train_cs.txt',
        'val_annotation': './data/ntu60/annotations/val_cs.txt',
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'checkpoint_name': 'best.pth',

        # System
        'num_workers': 4,
        'dry_run': False,
    }
