
import os
import sys
import json
import argparse
import subprocess
import numpy as np
from datetime import datetime


EXPERIMENTS = {
    'ntu60_cs': 'configs/experiments/ntu60_cs_joint.yaml',
    'ntu60_cv': 'configs/experiments/ntu60_cv_joint.yaml',
    'ntu120_csub': 'configs/experiments/ntu120_csub_joint.yaml',
    'ntu120_cset': 'configs/experiments/ntu120_cset_joint.yaml',
    'ucf101_s1': 'configs/experiments/ucf101_split1_joint.yaml',
}

PAPER_TARGETS = {
    'ntu60_cs': '93.2±0.3',
    'ntu60_cv': '96.8±0.2',
    'ntu120_csub': '88.1±0.4',
    'ntu120_cset': '89.0±0.3',
    'ucf101_3split': '95.8±0.4',
}


def run_one(config, seed, dry_run=False, epochs=None):
    """Run single training, return best accuracy."""
    cmd = [sys.executable, 'train.py', '--config', config, '--seed', str(seed)]
    if dry_run:
        cmd.append('--dry_run')
    if epochs:
        cmd.extend(['--epochs', str(epochs)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Best Val Acc' in line or 'Final Val Acc' in line:
            try:
                return float(line.split(':')[-1].strip().replace('%', ''))
            except ValueError:
                pass
    return 0.0


def run_experiment(name, config, seeds, dry_run=False, epochs=None):
    """Run one experiment N times."""
    print(f"\n{'='*60}")
    print(f"  {name}: {config}")
    print(f"{'='*60}")

    accs = []
    for i, seed in enumerate(seeds):
        print(f"  Run {i+1}/{len(seeds)} (seed={seed})...", end=" ", flush=True)
        acc = run_one(config, seed, dry_run, epochs)
        accs.append(acc)
        print(f"{acc:.2f}%")

    mean = np.mean(accs)
    std = np.std(accs, ddof=1) if len(accs) > 1 else 0
    target = PAPER_TARGETS.get(name, "N/A")
    print(f"  Result: {mean:.1f}±{std:.1f}%  (Paper: {target})")
    return {'mean': mean, 'std': std, 'accs': accs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', default=None,
                        choices=['ntu60', 'ntu120', 'ucf101', 'all'])
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.num_runs))
    exps = args.experiments or ['all']

    selected = {}
    if 'all' in exps or 'ntu60' in exps:
        selected['ntu60_cs'] = EXPERIMENTS['ntu60_cs']
        selected['ntu60_cv'] = EXPERIMENTS['ntu60_cv']
    if 'all' in exps or 'ntu120' in exps:
        selected['ntu120_csub'] = EXPERIMENTS['ntu120_csub']
        selected['ntu120_cset'] = EXPERIMENTS['ntu120_cset']
    if 'all' in exps or 'ucf101' in exps:
        selected['ucf101_s1'] = EXPERIMENTS['ucf101_s1']
        selected['ucf101_s2'] = EXPERIMENTS['ucf101_s2']
        selected['ucf101_s3'] = EXPERIMENTS['ucf101_s3']

    results = {}
    for name, config in selected.items():
        results[name] = run_experiment(name, config, seeds, args.dry_run, args.epochs)

    # UCF-101 3-split average
    ucf_accs = []
    for k in ['ucf101_s1', 'ucf101_s2', 'ucf101_s3']:
        if k in results:
            ucf_accs.extend(results[k]['accs'])
    if ucf_accs:
        print(f"\n  UCF-101 3-split avg: {np.mean(ucf_accs):.1f}±{np.std(ucf_accs,ddof=1):.1f}%")
        print(f"  Paper target:        {PAPER_TARGETS['ucf101_3split']}")

    # Save
    out_path = f"logs/experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('logs', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
