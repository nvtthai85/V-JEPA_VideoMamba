# Hybrid V-JEPA × VideoMamba for Video Action Recognition

> **Paper:** "Hybrid V-JEPA and VideoMamba Architecture for Robust Video Action Recognition"  
> **Venue:** LNCS Conference Proceedings (Scopus Q4)

Complete source code for reproducing all experiments in the paper.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Project Structure](#project-structure)
5. [Paper Results Summary](#paper-results-summary)

---

## Architecture Overview

```
Input (B, 3, 16, 224, 224)      ← RGB only, 16 frames
    │
    ├── Tubelet Embedding (Conv3D: t=2, p=16) → 1568 tokens + [CLS]
    │
    ├── VideoMamba Backbone (24 blocks, D=384, bidirectional SSM)
    │       Complexity: O(L) linear
    │       ├── Forward SSM scan  ─┐
    │       └── Backward SSM scan ─┘→ sum (Eq. 4)
    │
    ├──→ Classification Head (Attentive Probe)  → logits
    │       L_cls = CrossEntropy(logits, labels)
    │
    └──→ V-JEPA Predictor (4 Transformer blocks, dim=192)
            Complexity: O(n²) — predictor only, discarded at inference
            L_jepa = MSE(pred, EMA_target)
    
    Joint Loss: L = L_cls + α × L_jepa  (α=0.3)
```

**⚠ Complexity Note:** The backbone is O(L) linear, but the predictor uses O(n²) Transformer attention. The full training framework is NOT end-to-end linear. At inference, only the O(L) backbone runs.

---

## Environment Setup

### Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- ~40GB GPU memory (A100 recommended) or use `--batch_size 8` for smaller GPUs

### Installation

```bash
# 1. Clone and enter directory
cd hybrid_vjepa_videomamba

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation (dry run, no data/GPU needed)
python train.py --config configs/experiments/ntu60_cs_joint.yaml --dry_run --epochs 2
```

### VS Code Setup

Open the project folder in VS Code. Pre-configured launch configs are in `.vscode/launch.json`:
- **"Train: Dry Run"** — test full pipeline with synthetic data
- **"Train: NTU-60 CS"** — full training
- **"Ablation: Alpha"** — run alpha ablation

Press `F5` to start debugging with the selected configuration.

---

## Data Preparation

### NTU RGB+D 60/120

1. **Download** from [NTU ROSE Lab](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
   - Request access (academic use)
   - Download RGB videos (.avi files)

2. **Place files:**
   ```
   data/ntu60/videos/S001C001P001R001A001.avi
   data/ntu60/videos/S001C001P001R001A002.avi
   ...
   data/ntu120/videos/  (same format, 120 actions)
   ```

3. **Generate annotations:**
   ```bash
   python scripts/prepare_data.py --dataset ntu60
   python scripts/prepare_data.py --dataset ntu120
   ```

### UCF-101

1. **Download** from [UCF CRCV](https://www.crcv.ucf.edu/data/UCF101.php)
   - Download `UCF101.rar` and extract

2. **Place files** (organized by class):
   ```
   data/ucf101/split1/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
   data/ucf101/split1/ApplyLipstick/...
   ...
   ```

3. **Generate annotations:**
   ```bash
   python scripts/prepare_data.py --dataset ucf101
   ```

---

## Project Structure

```
hybrid_vjepa_videomamba/
├── .vscode/
│   ├── settings.json          # Python config
│   └── launch.json            # Debug configurations
├── configs/
│   ├── __init__.py            # Default hyperparameters (Table 1)
│   └── experiments/
│       ├── ntu60_cs_joint.yaml
│       ├── ntu60_cs_supervised.yaml
│       ├── ntu60_cv_joint.yaml
│       ├── ntu120_csub_joint.yaml
│       ├── ntu120_cset_joint.yaml
│       └── ucf101_split1_joint.yaml
├── models/
│   ├── mamba_block.py         # SelectiveSSM + MambaBlock + DropPath
│   ├── videomamba.py          # VideoMamba backbone (bidirectional)
│   ├── vjepa.py               # V-JEPA: predictor, EMA, masking
│   ├── heads.py               # CLS / MeanPool / AttentiveProbe
│   └── hybrid_model.py        # Complete hybrid model
├── datasets/
│   └── video_dataset.py       # VideoDataset + DummyDataset + loaders
├── engine/
│   └── trainer.py             # Training loop (AMP, EMA, early stop)
├── utils/
│   ├── metrics.py             # AverageMeter, accuracy
│   ├── scheduler.py           # CosineAnnealingWarmup
│   ├── checkpoint.py          # Save/load checkpoints
│   └── logger.py              # CSV + file logging
├── scripts/
│   └── prepare_data.py        # Generate annotation files
├── data/                      # Dataset directory (videos go here)
│   ├── ntu60/
│   ├── ntu120/
│   └── ucf101/
├── logs/                      # Training logs (.log + .csv)
├── checkpoints/               # Saved model weights
├── train.py                   # Main training entry point
├── evaluate.py                # Checkpoint evaluation
├── run_experiments.py         # Reproduce Tables 2-3
├── run_ablations.py           # Reproduce Tables 5-9
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Paper Results Summary

### Table 2: NTU RGB+D (RGB only, no external pretraining)

| Method           | NTU-60 CS     | NTU-60 CV     | NTU-120 CSub  | NTU-120 CSet  |
|------------------|---------------|---------------|---------------|---------------|
| **Ours (α=0.3)** | **93.2±0.3%** | **96.8±0.2%** | **88.1±0.4%** | **89.0±0.3%** |

### Table 3: UCF-101 (RGB only, no external pretraining)

| Method           | 3-Split Avg   |
|------------------|---------------|
| **Ours (α=0.3)** | **95.8±0.4%** |

### Table 4: Computational Cost

| Component        | GFLOPs | Params | Complexity |
|------------------|--------|--------|------------|
| Backbone         | ~340   | 26.0M  | O(L)       |
| V-JEPA Predictor | ~18    | 4.8M   | O(n²)      |
| Attentive Probe  | <1     | 3.4M   | O(L)       |
| **Inference**    | **~341** | **29.4M** | —       |
| **Training**     | **~359** | **34.2M** | —       |

### Training Configuration (Table 1)

| Parameter        | Value                              |
|------------------|------------------------------------|
| Optimizer        | AdamW (β₁=0.9, β₂=0.999)          |
| Learning rate    | 5×10⁻⁴ → 1×10⁻⁶ (cosine)         |
| Warmup           | 5 epochs                           |
| Weight decay     | 0.05                               |
| Epochs           | 100                                |
| Batch size       | 16                                 |
| AMP              | FP16                               |
| Seeds            | 42, 43, 44                         |

---

## License

This project is released for academic research purposes.
