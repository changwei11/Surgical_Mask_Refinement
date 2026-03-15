# Surgical Mask Refinement via RGB-Conditioned Latent Diffusion

Deep learning pipeline for refining surgical instrument segmentation masks using latent diffusion models conditioned on CLIP visual features.

## Overview

This project implements a **latent diffusion model** that refines coarse surgical instrument masks into high-quality refined masks. The model operates in a learned latent space (via VAE) and is conditioned on both **coarse mask latents** and **frozen CLIP RGB visual tokens**.

**Current Status:** Training pipeline complete through RGB-conditioned latent diffusion. Inference and evaluation coming next.

---

## Dataset

**Location:** `dataset/ece285_dataset/`

**Structure:**
```
dataset/ece285_dataset/
├── real_world/
│   ├── RGB/              # RGB surgical images
│   ├── coarse_mask/      # Coarse segmentation masks
│   └── refined_mask/     # High-quality refined masks (ground truth)
└── synthetic/
    ├── RGB/
    ├── coarse_mask/
    └── refined_mask/
```

**Splits:** Metadata in `data/metadata/`
- Train: ~9,057 samples
- Val: ~1,132 samples  
- Test: ~1,133 samples
- Total: ~11,322 samples (real_world + synthetic)

**Data Format:**
- RGB images: 700×493 → resized to 512×512
- Masks: Binary (0/1), resized to 512×512
- All images paired: `{RGB, coarse_mask, refined_mask}`

---

## Pipeline Architecture

### 1. Mask VAE

**Purpose:** Learn compressed latent representation of masks

**Architecture:**
- Input: Binary mask `[1, 512, 512]`
- Encoder: 4 downsampling blocks → latent `[8, 32, 32]`
- Decoder: 4 upsampling blocks → reconstruction `[1, 512, 512]`
- Loss: BCE + Dice + KL divergence (β=0.0001)

**Training Data:** Refined masks only

**Output:** Frozen encoder/decoder for latent diffusion

**Key Details:**
- `z = μ` (deterministic encoding, no sampling)
- Spatial latent: 16× compression (512×512 → 32×32)
- Checkpoint: `outputs/vae/checkpoints/best.pt` (~89MB)
- **Pretrained:** Available at `./checkpoints/vae_best.pt` (tracked in git)

### 2. CLIP Token Precomputation

**Purpose:** Extract frozen RGB visual features for conditioning

**Model:** `openai/clip-vit-base-patch16` (ViT-B/16@224px)

**Processing:**
1. Resize RGB: 512×512 → 224×224
2. Apply CLIP normalization
3. Extract patch tokens (CLS token removed)
4. Save to disk: `outputs/clip_tokens/{split}/{source}/{file_stem}.pt`

**Token Format:**
- Shape: `[196, 768]` per image
- 196 = 14×14 patch grid
- 768 = ViT-B/16 hidden dimension

**Storage:** ~200MB for full dataset

### 3. RGB-Conditioned Latent Diffusion

**Purpose:** Refine coarse masks to high-quality refined masks

**Inputs:**
- `z_coarse`: Encoded coarse mask `[8, 32, 32]` (frozen VAE)
- `z_refined`: Encoded refined mask `[8, 32, 32]` (training target)
- `rgb_tokens`: CLIP features `[196, 768]` (precomputed)

**Model Architecture:**
- U-Net backbone: 3 levels `[64, 128, 256]` channels
- Time embedding: Sinusoidal positional encoding
- Coarse conditioning: Concatenate `[z_t || z_coarse]` → `[16, 32, 32]` input
- RGB conditioning: Cross-attention with projected tokens at bottleneck
  - Token projection: 768D → 256D linear
  - 4-head cross-attention
- Parameters: ~6.1M trainable

**Training:**
- Objective: Epsilon-prediction MSE `||ε_θ(z_t, t, z_coarse, rgb_tokens) - ε||²`
- Scheduler: Linear β schedule, T=1000 steps
- Optimizer: AdamW, LR=1e-4
- Frozen: VAE encoder/decoder, CLIP tokenizer

**Checkpoints:** `outputs/diffusion_rgb/checkpoints/`

---

## Repository Structure

```
checkpoints/            # Pretrained model checkpoints (tracked in git)

configs/
├── model/              # Model architectures (VAE, diffusion, CLIP)
└── train/              # Training configs (overfit, full)

data/
├── metadata/           # Train/val/test splits (JSON)
├── dataset.py          # Paired dataset loader
└── token_dataset.py    # Token-aware dataset

models/
├── baselies
├── vae/                # Mask VAE, frozen interface
├── rgb/                # CLIP tokenizer
└── diffusion/          # Diffusion U-Net, scheduler, conditioner

trainers/
├── vae_trainer.py              # VAE training loop
├── diffusion_trainer.py        # Baseline diffusion
└── rgb_diffusion_trainer.py    # RGB-conditioned diffusion

scripts/
├── precompute_rgb_tokens.py            # Step 4
├── train_vae.py                        # Step 8
├── train_cvae.py
├── train_cgan.py
├── train_latent_diffusion.py           # Step 10 (baseline)
└── train_rgb_conditioned_diffusion.py  # Step 12

outputs/
├── vae/checkpoints/            # Trained VAE
├── clip_tokens/                # Precomputed tokens
├── diffusion/                  # Baseline diffusion (if trained)
└── diffusion_rgb/              # RGB-conditioned diffusion
```

---

## Training Commands

### Prerequisites

```bash
# Start Docker container
./start_container.sh

# Verify dataset structure
ls dataset/ece285_dataset/{real_world,synthetic}/{RGB,coarse_mask,refined_mask}
```

### Step 1: Precompute CLIP Tokens

```bash
python3 scripts/precompute_rgb_tokens.py \
    --config configs/model/rgb_tokenizer.yaml \
    --metadata_dir data/metadata \
    --output_dir outputs/clip_tokens \
    --device cuda \
    --batch_size 32
```

**Output:** `outputs/clip_tokens/{train,val,test}/{real_world,synthetic}/*.pt`

### Step 2: Train VAE(already finished, i will upload the checkpoints)

**Overfit sanity check (recommended first):**
```bash
bash train_vae_overfit.sh
```

**Full training:**
```bash
bash train_vae.sh
```

Or manually:
```bash
python3 scripts/train_vae.py \
    --config configs/train/vae_train.yaml \
    --device cuda \
    --epochs 50 \
    --run_name vae_full
```

**Output:** `outputs/vae/checkpoints/best.pt`

### Step 3: Train RGB-Conditioned Diffusion

**Full training:**
```bash
bash train_rgb_conditioned_diffusion.sh
```

Or manually:
```bash
python3 scripts/train_rgb_conditioned_diffusion.py \
    --train_config configs/train/diffusion_rgb_train.yaml \
    --vae_config configs/model/vae.yaml \
    --diffusion_config configs/model/diffusion_rgb.yaml \
    --vae_checkpoint outputs/vae/checkpoints/best.pt \
    --device cuda \
    --epochs 500 \
    --eval_every_n_epochs 50 \
    --save_every_n_epochs 50 \
    --run_name rgb_conditioned_full
```

**Output:** `outputs/diffusion_rgb/checkpoints/best.pt`

---

### Step 4: Run Baseline Diffusion Inference (with optional test-time augmentation)

```bash
python3 scripts/infer_diffusion.py --config configs/infer/diffusion_infer.yaml --split test --source all
```


### Step 5: Train Baselines
```
python3 train_cvae.py --dataset_type real_world --epochs 500
```
```
python3 train_cgan.py --dataset_type real_world --epochs 500
```

---

## Inference Visualization

Each visualization image has 4 panels:
1. RGB input
2. coarse mask
3. predicted refined mask (with per-sample Dice/IoU in title)
4. refined ground truth

You can quickly inspect a few examples with:

```bash
python3 - <<'PY'
from pathlib import Path
vis_dir = Path('outputs/inference_test_aug/visualizations')
for p in sorted(vis_dir.glob('*_comparison.png'))[:10]:
    print(p)
PY
```

## Monitoring Training

### Tmux Sessions

Training runs in detached tmux sessions:

```bash
# List active sessions
tmux ls

# Attach to training
tmux attach -t rgb_conditioned_diffusion

# Detach (while inside)
Ctrl+B, then D
```

### WandB Logging

All training logged to WandB (online mode):
- **VAE Project:** `surgical-mask-vae`
- **Diffusion Project:** `surgical-rgb-conditioned-diffusion`

Metrics:
- `train/loss`, `val/loss`
- Latent statistics
- Reconstruction visualizations (validation)

### Logs

```bash
# VAE
tail -f outputs/vae/logs/*.log

# RGB-conditioned diffusion
tail -f outputs/diffusion_rgb/logs/*.log
```

---

## Output Structure

```
checkpoints/            # Pretrained checkpoints (tracked in git repo)
├── vae_best.pt         # Pretrained VAE checkpoint (~89MB)
└── ...                 # Other pretrained models

outputs/                # Training outputs (gitignored)
├── vae/
│   ├── checkpoints/
│   │   ├── best.pt         # Lowest val loss
│   │   └── latest.pt
│   └── visualizations/
│       └── epoch_*.png     # Reconstruction grids
│
├── clip_tokens/
│   ├── train/
│   │   ├── real_world/*.pt
│   │   └── synthetic/*.pt
│   ├── val/
│   └── test/
│
└── diffusion_rgb/
    ├── checkpoints/
    │   ├── best.pt         # Lowest val loss
    │   ├── latest.pt
    │   └── epoch_0050.pt, epoch_0100.pt, ...
    └── visualizations/
        └── epoch_*.png     # 4-column: coarse GT | refined GT | decoded coarse | prediction
```

**Note:** The `./checkpoints/` folder at the repository root contains pretrained model weights that are tracked in git for easy access. Training outputs are saved to `./outputs/` and are gitignored.

---

## Key Implementation Details

### VAE Latent Interface

```python
# Frozen VAE used during diffusion training
vae = FrozenVAELatentInterface(
    model_config_path="configs/model/vae.yaml",
    checkpoint_path="outputs/vae/checkpoints/best.pt",
    use_mu_only=True  # Deterministic z = μ
)

# Encoding (no gradients)
with torch.no_grad():
    z_coarse = vae.encode_coarse_mask(coarse_mask)   # [B, 8, 32, 32]
    z_refined = vae.encode_refined_mask(refined_mask) # [B, 8, 32, 32]
```

### Diffusion Training Step

```python
# 1. Encode masks (frozen VAE)
z_coarse = vae.encode_coarse_mask(coarse_mask)
z_refined = vae.encode_refined_mask(refined_mask)

# 2. Sample timestep and noise
t ~ Uniform(0, T-1)
ε ~ N(0, I)

# 3. Forward diffusion
z_t = sqrt(α_t) * z_refined + sqrt(1 - α_t) * ε

# 4. Predict noise (trainable)
ε_θ = model(z_t, t, z_coarse, rgb_tokens)

# 5. Compute loss
loss = MSE(ε_θ, ε)
```

---


## Troubleshooting

**Token files missing:**
```bash
# Run Step 4 first
python3 scripts/precompute_rgb_tokens.py ...
```

**VAE checkpoint missing:**
```bash
# Train VAE first
bash train_vae.sh
```

**CUDA OOM:**
```bash
# Reduce batch size in config
--batch_size 8
```

**WandB login:**
```bash
wandb login
# Paste API key
```
