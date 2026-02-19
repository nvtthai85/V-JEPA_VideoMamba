
import argparse
import yaml
import torch
from torch.cuda.amp import autocast
from configs import get_default_config
from models import HybridVJEPAVideoMamba
from datasets.video_dataset import build_dataloaders
from utils.metrics import AverageMeter, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    config = get_default_config()
    with open(args.config) as f:
        config.update(yaml.safe_load(f) or {})
    config['dry_run'] = args.dry_run

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu')

    model = HybridVJEPAVideoMamba(
        num_classes=config['num_classes'],
        bidirectional=config.get('bidirectional', True),
        head_type=config.get('head_type', 'attentive_probe'),
        alpha=config.get('alpha', 0.3),
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model = model.to(device).eval()

    _, val_loader = build_dataloaders(config)

    acc_m = AverageMeter()
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            with autocast(enabled=device.type == 'cuda'):
                logits = model.inference(videos)
            acc1 = accuracy(logits, labels, topk=(1,))[0]
            acc_m.update(acc1, videos.size(0))

    print(f"Validation Accuracy: {acc_m.avg:.2f}%")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == '__main__':
    main()
