"""Checkpoint save/load utilities."""
import os
import torch
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(state, filepath, is_best=False):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    torch.save(state, filepath)
    if is_best:
        best_path = filepath.replace('.pth', '_best.pth')
        torch.save(state, best_path)
    logger.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """Load checkpoint and restore model/optimizer state."""
    if not os.path.isfile(filepath):
        logger.warning(f"No checkpoint at {filepath}")
        return 0, 0.0
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt.get('state_dict', {})))
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    epoch = ckpt.get('epoch', 0)
    best_acc = ckpt.get('best_acc', 0.0)
    logger.info(f"Loaded checkpoint: {filepath} (epoch {epoch}, best_acc {best_acc:.2f}%)")
    return epoch, best_acc
