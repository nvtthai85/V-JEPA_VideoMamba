import os
import sys
import argparse
import subprocess
import json
import numpy as np
from datetime import datetime


def run_train(extra_args, seeds, dry_run=False, epochs=None):
    """Run train.py with extra args, return mean±std accuracy."""
    base_config = 'configs/experiments/ntu60_cs_joint.yaml'
    accs = []
    for seed in seeds:
        cmd = [sys.executable, 'train.py', '--config', base_config, '--seed', str(seed)]
        cmd.extend(extra_args)
        if dry_run:
            cmd.append('--dry_run')
        if epochs:
            cmd.extend(['--epochs', str(epochs)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'Best Val Acc' in line or 'Final Val Acc' in line:
                try:
                    accs.append(float(line.split(':')[-1].strip().replace('%', '')))
                except ValueError:
                    pass
                break
    if not accs:
        return 0.0, 0.0
    return np.mean(accs), np.std(accs, ddof=1) if len(accs) > 1 else 0.0


def ablation_alpha(seeds, dry_run, epochs):
    """Table 6: α sensitivity (fine-grained around 0.3)."""
    print("\n" + "="*60)
    print("  ABLATION: Alpha (α) — Paper Table 6")
    print("="*60)
    alphas = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.5, 0.7, 1.0]
    for a in alphas:
        mean, std = run_train(['--alpha', str(a)], seeds, dry_run, epochs)
        print(f"  α={a:.2f}: {mean:.1f}±{std:.1f}%")


def ablation_scan(seeds, dry_run, epochs):
    """Table 7: scanning direction."""
    print("\n" + "="*60)
    print("  ABLATION: Scanning Direction — Paper Table 7")
    print("="*60)
    # Note: forward-only and backward-only need code change (bidirectional=false)
    # Here we test bidirectional vs unidirectional
    for bidir in [True, False]:
        label = "Bidirectional" if bidir else "Unidirectional"
        mean, std = run_train(['--bidirectional', str(bidir)], seeds, dry_run, epochs)
        print(f"  {label}: {mean:.1f}±{std:.1f}%")


def ablation_head(seeds, dry_run, epochs):
    """Table 8: classification head."""
    print("\n" + "="*60)
    print("  ABLATION: Classification Head — Paper Table 8")
    print("="*60)
    for head in ['cls_token', 'mean_pool', 'attentive_probe']:
        mean, std = run_train(['--head_type', head], seeds, dry_run, epochs)
        print(f"  {head:20s}: {mean:.1f}±{std:.1f}%")


def ablation_mask_ratio(seeds, dry_run, epochs):
    """Table 9: mask ratio (ρ)."""
    print("\n" + "="*60)
    print("  ABLATION: Mask Ratio (ρ) — Paper Table 9")
    print("="*60)
    ratios = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    for r in ratios:
        mean, std = run_train(['--mask_ratio', str(r)], seeds, dry_run, epochs)
        print(f"  ρ={r:.2f}: {mean:.1f}±{std:.1f}%")


def ablation_training_mode(seeds, dry_run, epochs):
    """Table 5: training mode comparison."""
    print("\n" + "="*60)
    print("  ABLATION: Training Mode — Paper Table 5")
    print("="*60)
    # Supervised only
    mean, std = run_train(['--alpha', '0.0'], seeds, dry_run, epochs)
    print(f"  Supervised (α=0):    {mean:.1f}±{std:.1f}%")
    # Joint
    mean, std = run_train(['--alpha', '0.3'], seeds, dry_run, epochs)
    print(f"  Joint (α=0.3):       {mean:.1f}±{std:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, required=True,
                        choices=['alpha', 'scan', 'head', 'mask_ratio',
                                 'training_mode', 'all'])
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.num_runs))

    ablations = {
        'alpha': ablation_alpha,
        'scan': ablation_scan,
        'head': ablation_head,
        'mask_ratio': ablation_mask_ratio,
        'training_mode': ablation_training_mode,
    }

    if args.ablation == 'all':
        for name, fn in ablations.items():
            fn(seeds, args.dry_run, args.epochs)
    else:
        ablations[args.ablation](seeds, args.dry_run, args.epochs)


if __name__ == '__main__':
    main()
