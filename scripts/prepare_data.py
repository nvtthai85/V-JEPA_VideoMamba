#!/usr/bin/env python3
"""
prepare_data.py — Generate annotation files for NTU RGB+D and UCF-101.

This script creates train/val annotation files from downloaded video datasets.
Each annotation file has one line per sample: "relative/path/to/video.avi label"

STEP 1: Download datasets manually:
    - NTU RGB+D 60:  https://rose1.ntu.edu.sg/dataset/actionRecognition/
    - NTU RGB+D 120: https://rose1.ntu.edu.sg/dataset/actionRecognition/
    - UCF-101:       https://www.crcv.ucf.edu/data/UCF101.php

STEP 2: Place videos:
    data/ntu60/videos/     — all .avi files from NTU-60
    data/ntu120/videos/    — all .avi files from NTU-120
    data/ucf101/split1/    — UCF-101 organized by class folders

STEP 3: Run this script:
    python scripts/prepare_data.py --dataset ntu60
    python scripts/prepare_data.py --dataset ntu120
    python scripts/prepare_data.py --dataset ucf101
    python scripts/prepare_data.py --dataset all

Output:
    data/{dataset}/annotations/train_{split}.txt
    data/{dataset}/annotations/val_{split}.txt
"""

import os
import re
import argparse
import random


# NTU RGB+D 60 Cross-Subject training subjects
NTU_CS_TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16,
    17, 18, 19, 25, 27, 28, 31, 34, 35, 38
}

# NTU RGB+D 60 Cross-View training cameras
NTU_CV_TRAIN_CAMERAS = {2, 3}


def parse_ntu_filename(fname):
    """Parse NTU filename: SsssCcccPpppRrrrAaaa.avi
    S=setup, C=camera, P=subject, R=replication, A=action
    """
    m = re.match(r'S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)', fname)
    if m:
        return {
            'setup': int(m.group(1)),
            'camera': int(m.group(2)),
            'subject': int(m.group(3)),
            'replication': int(m.group(4)),
            'action': int(m.group(5)) - 1,  # 0-indexed
        }
    return None


def prepare_ntu60(data_root='data/ntu60'):
    """Generate NTU-60 annotation files for CS and CV protocols."""
    video_dir = os.path.join(data_root, 'videos')
    ann_dir = os.path.join(data_root, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)

    if not os.path.isdir(video_dir):
        print(f"  [SKIP] {video_dir} not found. Download NTU-60 first.")
        return

    videos = sorted([f for f in os.listdir(video_dir) if f.endswith('.avi')])
    print(f"  Found {len(videos)} videos in {video_dir}")

    cs_train, cs_val = [], []
    cv_train, cv_val = [], []

    for v in videos:
        info = parse_ntu_filename(v)
        if not info or info['action'] >= 60:
            continue
        line = f"videos/{v} {info['action']}"
        # Cross-Subject
        if info['subject'] in NTU_CS_TRAIN_SUBJECTS:
            cs_train.append(line)
        else:
            cs_val.append(line)
        # Cross-View
        if info['camera'] in NTU_CV_TRAIN_CAMERAS:
            cv_train.append(line)
        else:
            cv_val.append(line)

    for name, data in [('train_cs', cs_train), ('val_cs', cs_val),
                       ('train_cv', cv_train), ('val_cv', cv_val)]:
        path = os.path.join(ann_dir, f'{name}.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(data))
        print(f"  Wrote {path}: {len(data)} samples")


def prepare_ntu120(data_root='data/ntu120'):
    """Generate NTU-120 annotations for Cross-Subject and Cross-Setup."""
    video_dir = os.path.join(data_root, 'videos')
    ann_dir = os.path.join(data_root, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)

    if not os.path.isdir(video_dir):
        print(f"  [SKIP] {video_dir} not found. Download NTU-120 first.")
        return

    videos = sorted([f for f in os.listdir(video_dir) if f.endswith('.avi')])
    print(f"  Found {len(videos)} videos in {video_dir}")

    # NTU-120 Cross-Subject: subjects 1-53 train, 54-106 test
    train_subjects_120 = set(range(1, 54))
    # NTU-120 Cross-Setup: even setups train, odd setups test
    csub_train, csub_val = [], []
    cset_train, cset_val = [], []

    for v in videos:
        info = parse_ntu_filename(v)
        if not info:
            continue
        line = f"videos/{v} {info['action']}"
        if info['subject'] in train_subjects_120:
            csub_train.append(line)
        else:
            csub_val.append(line)
        if info['setup'] % 2 == 0:
            cset_train.append(line)
        else:
            cset_val.append(line)

    for name, data in [('train_csub', csub_train), ('val_csub', csub_val),
                       ('train_cset', cset_train), ('val_cset', cset_val)]:
        path = os.path.join(ann_dir, f'{name}.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(data))
        print(f"  Wrote {path}: {len(data)} samples")


def prepare_ucf101(data_root='data/ucf101'):
    """Generate UCF-101 annotations for 3 splits."""
    ann_dir = os.path.join(data_root, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)

    # Check for class-organized directory
    for split in [1, 2, 3]:
        split_dir = os.path.join(data_root, f'split{split}')
        if not os.path.isdir(split_dir):
            # Try flat video directory
            print(f"  [INFO] {split_dir} not found.")
            print(f"  Place UCF-101 videos in data/ucf101/split{{1,2,3}}/ organized by class folders")
            print(f"  Or provide annotation files directly in {ann_dir}/")
            continue

        classes = sorted(os.listdir(split_dir))
        class_map = {c: i for i, c in enumerate(classes)}

        train_list, test_list = [], []
        for cls_name in classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            videos = sorted([f for f in os.listdir(cls_dir) if f.endswith(('.avi', '.mp4'))])
            label = class_map[cls_name]
            # 70/30 split (approximate UCF standard)
            n_train = int(len(videos) * 0.7)
            for v in videos[:n_train]:
                train_list.append(f"split{split}/{cls_name}/{v} {label}")
            for v in videos[n_train:]:
                test_list.append(f"split{split}/{cls_name}/{v} {label}")

        for name, data in [(f'trainlist0{split}', train_list),
                           (f'testlist0{split}', test_list)]:
            path = os.path.join(ann_dir, f'{name}.txt')
            with open(path, 'w') as f:
                f.write('\n'.join(data))
            print(f"  Wrote {path}: {len(data)} samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ntu60', 'ntu120', 'ucf101', 'all'],
                        default='all')
    args = parser.parse_args()

    print("\n  Data Preparation for Hybrid V-JEPA × VideoMamba\n")

    if args.dataset in ('ntu60', 'all'):
        print("[NTU RGB+D 60]")
        prepare_ntu60()
    if args.dataset in ('ntu120', 'all'):
        print("\n[NTU RGB+D 120]")
        prepare_ntu120()
    if args.dataset in ('ucf101', 'all'):
        print("\n[UCF-101]")
        prepare_ucf101()

    print("\nDone! Check data/*/annotations/ for generated files.")


if __name__ == '__main__':
    main()
