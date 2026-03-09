# Hybrid Training Setup

## Overview

This setup trains on:
- **Training**: ALL synthetic data (3,322) + real_world 0-1999 (2,000) = **5,322 samples**
- **Validation**: real_world 2000-5999 = **4,000 samples**
- **Test**: real_world 6000-7999 = **2,000 samples**

## Files Created

### Metadata
- `data/metadata/hybrid_train.json` - Training split metadata
- `data/metadata/hybrid_val.json` - Validation split metadata
- `data/metadata/hybrid_test.json` - Test split metadata

### Configuration
- `configs/train/diffusion_rgb_hybrid.yaml` - Training configuration for hybrid setup

### Scripts
- `train_rgb_conditioned_diffusion_hybrid.sh` - Shell script to launch hybrid training
- `scripts/create_hybrid_split.py` - Script that created the hybrid splits
- `scripts/setup_hybrid_tokens.py` - Script that set up token symlinks

### Token Symlinks
- `outputs/clip_tokens/hybrid_train/` → Links to train split tokens
- `outputs/clip_tokens/hybrid_val/` → Links to val split tokens
- `outputs/clip_tokens/hybrid_test/` → Links to test split tokens

## How to Use

### 1. Launch Hybrid Training

```bash
bash train_rgb_conditioned_diffusion_hybrid.sh
```

This will:
- Create a tmux session named `rgb_diffusion_hybrid`
- Train for 500 epochs
- Save checkpoints to `outputs/diffusion_rgb_hybrid/`
- Log to WandB with tags `["hybrid-training", "rgb-conditioned", "synthetic+real_0-1999"]`

### 2. Monitor Training

Attach to tmux session:
```bash
tmux attach -t rgb_diffusion_hybrid
```

View logs:
```bash
tail -f outputs/diffusion_rgb_hybrid/logs/train_rgb_hybrid.log
```

Check WandB:
- Project: `surgical-rgb-conditioned-diffusion`
- Run name: `rgb_hybrid_synthetic_real0-1999`

### 3. Manage Training Session

List tmux sessions:
```bash
tmux ls
```

Kill training session:
```bash
tmux kill-session -t rgb_diffusion_hybrid
```

## Data Split Details

### Training Data (5,322 samples)
- Synthetic: 3,322 samples (100% of synthetic data)
- Real world: 2,000 samples (stems 0-1999)

### Validation Data (4,000 samples)
- Real world: 4,000 samples (stems 2000-5999)

### Test Data (2,000 samples)
- Real world: 2,000 samples (stems 6000-7999)

## Augmentation

Augmentation is applied to training data using `configs/train/augmentation.yaml`:
- 50% of samples get augmented
- Operations: erosion, dilation, edge blobs, drop parts, cutout
- Only applied to coarse masks (refined masks stay unchanged)

## Output Structure

```
outputs/diffusion_rgb_hybrid/
├── checkpoints/
│   ├── best.pt          # Best validation loss
│   ├── latest.pt        # Most recent checkpoint
│   └── epoch_*.pt       # Milestone checkpoints
├── visualizations/
│   └── epoch_*.png      # Visualization grids
└── logs/
    └── train_rgb_hybrid.log  # Training log
```

## Comparison with Full Training

| Setup | Train Samples | Val Samples | Real/Synthetic Mix |
|-------|--------------|-------------|-------------------|
| **Full** | 9,057 | 1,132 | 6,400 real + 2,657 synthetic |
| **Hybrid** | 5,322 | 4,000 | 2,000 real + 3,322 synthetic |

The hybrid setup:
- Uses ALL synthetic data (better synthetic representation)
- Uses limited real data for training (0-1999)
- Tests generalization on unseen real data (2000+)
