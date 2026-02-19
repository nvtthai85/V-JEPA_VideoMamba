"""
Video Dataset for NTU RGB+D 60/120 and UCF-101.

Supports:
    - Annotation-file based loading (path label format)
    - Uniform temporal sampling with jitter
    - Spatial augmentation (RandomResizedCrop, HFlip, ColorJitter)
    - ImageNet normalization
    - decord (fast) or OpenCV fallback

Directory layout expected:
    data/
    ├── ntu60/
    │   ├── videos/         # .avi or .mp4 files
    │   └── annotations/    # train_cs.txt, val_cs.txt, etc.
    ├── ntu120/
    │   ├── videos/
    │   └── annotations/
    └── ucf101/
        ├── split1/         # or videos/ with annotation files

Annotation format (each line):
    path/to/video.avi 42
    (space-separated: relative_path label_int)
"""

import os
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logger = logging.getLogger(__name__)

try:
    import decord
    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ImageNet normalization (paper Table 1)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VideoDataset(Dataset):
    """Generic video dataset for action recognition.

    Args:
        annotation_file: Path to txt file with "video_path label" per line.
        data_root:       Root directory for video files.
        num_frames:      Frames to sample per clip. Default 16.
        img_size:        Spatial crop size. Default 224.
        is_train:        Training mode (applies augmentation). Default True.
        temporal_jitter:  Add random jitter to frame indices. Default True.
    """

    def __init__(
        self,
        annotation_file,
        data_root="",
        num_frames=16,
        img_size=224,
        is_train=True,
        temporal_jitter=True,
    ):
        self.data_root = data_root
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_train = is_train
        self.temporal_jitter = temporal_jitter and is_train

        # Load annotations
        self.samples = []
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    path = parts[0]
                    label = int(parts[-1])
                    full_path = os.path.join(data_root, path) if data_root else path
                    self.samples.append((full_path, label))

        logger.info(f"Loaded {len(self.samples)} samples from {annotation_file}")

        # Transforms
        if is_train:
            self.spatial_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4
                ),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            frames = self._load_video(path)
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}. Using random replacement.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        # frames: (T, C, H, W) float [0, 1]
        frames = self._apply_spatial(frames)   # (T, C, img_size, img_size)
        video = frames.permute(1, 0, 2, 3)    # (C, T, H, W) — PyTorch video format

        return video, label

    def _load_video(self, path):
        """Load and temporally sample video frames."""
        if HAS_DECORD:
            return self._load_decord(path)
        elif HAS_CV2:
            return self._load_cv2(path)
        else:
            raise RuntimeError(
                "Neither decord nor opencv installed. "
                "Install: pip install decord  or  pip install opencv-python"
            )

    def _load_decord(self, path):
        vr = decord.VideoReader(path, num_threads=1)
        total = len(vr)
        indices = self._sample_indices(total)
        frames = vr.get_batch(indices)           # (T, H, W, C) uint8 tensor
        frames = frames.float() / 255.0
        frames = frames.permute(0, 3, 1, 2)     # (T, C, H, W)
        return frames

    def _load_cv2(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 300  # fallback
        indices = self._sample_indices(total)

        frames = []
        for idx in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(frame).float() / 255.0)
            else:
                # Repeat last frame
                frames.append(frames[-1] if frames else torch.zeros(224, 224, 3))
        cap.release()

        frames = torch.stack(frames)             # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2)     # (T, C, H, W)
        return frames

    def _sample_indices(self, total_frames):
        """Uniform temporal sampling with optional jitter."""
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
            return indices[:self.num_frames]

        stride = total_frames / self.num_frames
        indices = []
        for i in range(self.num_frames):
            center = int(i * stride + stride / 2)
            if self.temporal_jitter:
                jitter = random.randint(-int(stride / 4), int(stride / 4))
                center = max(0, min(total_frames - 1, center + jitter))
            indices.append(center)
        return indices

    def _apply_spatial(self, frames):
        """Apply spatial transforms consistently across all frames."""
        T = frames.shape[0]

        if self.is_train:
            # Random crop parameters (same for all frames)
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                frames[0], scale=(0.5, 1.0), ratio=(3/4, 4/3)
            )
            flip = random.random() < 0.5
            jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
            jitter_fn = jitter.forward  # deterministic per-clip

            out = []
            for t in range(T):
                f = transforms.functional.resized_crop(
                    frames[t], i, j, h, w, [self.img_size, self.img_size]
                )
                if flip:
                    f = transforms.functional.hflip(f)
                f = jitter_fn(f)
                f = transforms.functional.normalize(f, IMAGENET_MEAN, IMAGENET_STD)
                out.append(f)
        else:
            size = int(self.img_size * 1.15)
            out = []
            for t in range(T):
                f = transforms.functional.resize(frames[t], [size, size])
                f = transforms.functional.center_crop(f, [self.img_size, self.img_size])
                f = transforms.functional.normalize(f, IMAGENET_MEAN, IMAGENET_STD)
                out.append(f)

        return torch.stack(out)


class DummyVideoDataset(Dataset):
    """Synthetic dataset for testing/debugging (no real videos needed).

    Generates random tensors with correct shapes.
    Use with --dry_run for verifying pipeline before running on real data.
    """

    def __init__(self, num_samples=1000, num_classes=60, num_frames=16, img_size=224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        video = torch.randn(3, self.num_frames, self.img_size, self.img_size)
        label = idx % self.num_classes
        return video, label


def build_dataloaders(config, is_distributed=False):
    """Build train/val DataLoaders from config dict.

    Config keys used:
        data_root, train_annotation, val_annotation,
        num_frames, img_size, batch_size, num_workers, dry_run, num_classes
    """
    if config.get('dry_run', False):
        logger.info("DRY RUN: using synthetic data")
        train_ds = DummyVideoDataset(
            num_samples=config.get('batch_size', 16) * 4,
            num_classes=config['num_classes'],
            num_frames=config.get('num_frames', 16),
            img_size=config.get('img_size', 224),
        )
        val_ds = DummyVideoDataset(
            num_samples=config.get('batch_size', 16) * 2,
            num_classes=config['num_classes'],
            num_frames=config.get('num_frames', 16),
            img_size=config.get('img_size', 224),
        )
    else:
        train_ds = VideoDataset(
            annotation_file=config['train_annotation'],
            data_root=config.get('data_root', ''),
            num_frames=config.get('num_frames', 16),
            img_size=config.get('img_size', 224),
            is_train=True,
        )
        val_ds = VideoDataset(
            annotation_file=config['val_annotation'],
            data_root=config.get('data_root', ''),
            num_frames=config.get('num_frames', 16),
            img_size=config.get('img_size', 224),
            is_train=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )

    return train_loader, val_loader
