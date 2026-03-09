#!/bin/bash
# ==============================================================================
# RGB-Conditioned Latent Diffusion HYBRID Training Script
# ==============================================================================
# This script launches hybrid RGB-conditioned latent diffusion training in tmux.
# Training data: ALL synthetic + real_world 0-1999
# Val/Test data: real_world 2000+
#
# Usage:
#   bash train_rgb_conditioned_diffusion_hybrid.sh
# ==============================================================================

set -euo pipefail

# ==============================================================================
# Configuration Variables (Edit these as needed)
# ==============================================================================

SESSION_NAME="rgb_diffusion_hybrid"
PROJECT_ROOT="/workspace/ece285"
TRAIN_CONFIG="configs/train/diffusion_rgb_hybrid.yaml"
VAE_CONFIG="configs/model/vae.yaml"
DIFFUSION_CONFIG="configs/model/diffusion_rgb.yaml"
VAE_CHECKPOINT="outputs/vae/checkpoints/best.pt"
DEVICE="cuda"
RUN_NAME="rgb_hybrid_synthetic_real0-1999"
OUTPUT_DIR="outputs/diffusion_rgb_hybrid"
EPOCHS="500"
EVAL_EVERY="10"
SAVE_EVERY="10"
LOG_DIR="outputs/diffusion_rgb_hybrid/logs"
LOG_FILE="$LOG_DIR/train_rgb_hybrid.log"
RESUME_CKPT=""  # Set to checkpoint path to resume
EXTRA_ARGS=""   # Additional CLI arguments if needed

# ==============================================================================
# Pre-flight Checks
# ==============================================================================

echo "========================================================================"
echo "RGB-CONDITIONED HYBRID LATENT DIFFUSION TRAINING LAUNCHER"
echo "========================================================================"
echo ""
echo "Training data: ALL synthetic + real_world 0-1999 (5,322 samples)"
echo "Val data: real_world 2000-5999 (4,000 samples)"
echo "Test data: real_world 6000-7999 (2,000 samples)"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "ERROR: tmux is not installed."
    echo "Please install tmux: apt-get install tmux"
    exit 1
fi

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not available."
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT" || exit 1
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if config files exist
echo "Checking configuration files..."
for config_file in "$TRAIN_CONFIG" "$VAE_CONFIG" "$DIFFUSION_CONFIG"; do
    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config file not found: $config_file"
        exit 1
    fi
    echo "  ✓ $config_file"
done
echo ""

# Check if hybrid metadata exists
echo "Checking hybrid metadata files..."
for split in "hybrid_train" "hybrid_val" "hybrid_test"; do
    metadata_file="data/metadata/${split}.json"
    if [ ! -f "$metadata_file" ]; then
        echo "ERROR: Hybrid metadata not found: $metadata_file"
        echo "Please run: python3 scripts/create_hybrid_split.py"
        exit 1
    fi
    echo "  ✓ $metadata_file"
done
echo ""

# Check if VAE checkpoint exists
echo "Checking VAE checkpoint..."
if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "ERROR: VAE checkpoint not found: $VAE_CHECKPOINT"
    echo "Please train the VAE first."
    exit 1
fi
echo "  ✓ $VAE_CHECKPOINT ($(du -h $VAE_CHECKPOINT | cut -f1))"
echo ""

# Check if token directory exists
echo "Checking precomputed CLIP tokens..."
TOKEN_DIR="outputs/clip_tokens"
if [ ! -d "$TOKEN_DIR" ]; then
    echo "ERROR: Token directory not found: $TOKEN_DIR"
    echo "Please run token precomputation first."
    exit 1
fi
echo "  ✓ $TOKEN_DIR"
echo ""

# Create output and log directories if they don't exist
echo "Creating output directories..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
echo "  ✓ $OUTPUT_DIR"
echo "  ✓ $LOG_DIR"
echo ""

# ==============================================================================
# Check if tmux session already exists
# ==============================================================================

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "========================================================================"
    echo "WARNING: Tmux session '$SESSION_NAME' already exists!"
    echo "========================================================================"
    echo ""
    echo "The training session is already running."
    echo ""
    echo "To attach to the existing session:"
    echo "  tmux attach -t $SESSION_NAME"
    echo ""
    echo "To view logs:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "To kill the existing session and start fresh:"
    echo "  tmux kill-session -t $SESSION_NAME"
    echo "  bash $0"
    echo ""
    exit 0
fi

# ==============================================================================
# Build Training Command
# ==============================================================================

TRAINING_CMD="python3 scripts/train_rgb_conditioned_diffusion.py"
TRAINING_CMD="$TRAINING_CMD --train_config $TRAIN_CONFIG"
TRAINING_CMD="$TRAINING_CMD --vae_config $VAE_CONFIG"
TRAINING_CMD="$TRAINING_CMD --diffusion_config $DIFFUSION_CONFIG"
TRAINING_CMD="$TRAINING_CMD --vae_checkpoint $VAE_CHECKPOINT"
TRAINING_CMD="$TRAINING_CMD --device $DEVICE"
TRAINING_CMD="$TRAINING_CMD --epochs $EPOCHS"
TRAINING_CMD="$TRAINING_CMD --eval_every_n_epochs $EVAL_EVERY"
TRAINING_CMD="$TRAINING_CMD --save_every_n_epochs $SAVE_EVERY"
TRAINING_CMD="$TRAINING_CMD --run_name $RUN_NAME"
TRAINING_CMD="$TRAINING_CMD --output_dir $OUTPUT_DIR"

# Add resume checkpoint if specified
if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
    TRAINING_CMD="$TRAINING_CMD --resume $RESUME_CKPT"
    echo "Resuming from checkpoint: $RESUME_CKPT"
fi

# Add extra arguments if specified
if [ -n "$EXTRA_ARGS" ]; then
    TRAINING_CMD="$TRAINING_CMD $EXTRA_ARGS"
fi

# Log the command with tee
FULL_CMD="$TRAINING_CMD 2>&1 | tee $LOG_FILE"

# ==============================================================================
# Launch Training in Tmux
# ==============================================================================

echo "========================================================================"
echo "LAUNCHING RGB-CONDITIONED HYBRID LATENT DIFFUSION TRAINING"
echo "========================================================================"
echo ""
echo "Session name: $SESSION_NAME"
echo "Run name: $RUN_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Epochs: $EPOCHS"
echo "Eval frequency: Every $EVAL_EVERY epochs"
echo "Save frequency: Every $SAVE_EVERY epochs"
echo ""
echo "Training command:"
echo "  $TRAINING_CMD"
echo ""
echo "------------------------------------------------------------------------"

# Export WANDB_MODE to online
export WANDB_MODE=online

# Create tmux session and run training
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT" bash -c "
    export WANDB_MODE=online
    cd $PROJECT_ROOT
    echo '========================================================================'
    echo 'RGB-CONDITIONED HYBRID LATENT DIFFUSION TRAINING'
    echo 'Session: $SESSION_NAME'
    echo 'Started: \$(date)'
    echo '========================================================================'
    echo ''
    $FULL_CMD
    echo ''
    echo '========================================================================'
    echo 'TRAINING FINISHED'
    echo 'Ended: \$(date)'
    echo '========================================================================'
    echo ''
    echo 'Press ENTER to close this tmux session, or press Ctrl+B then D to detach.'
    read
"

# ==============================================================================
# Success Message
# ==============================================================================

echo "✓ Training launched successfully in tmux session: $SESSION_NAME"
echo ""
echo "========================================================================"
echo "NEXT STEPS"
echo "========================================================================"
echo ""
echo "1. Attach to the training session:"
echo "     tmux attach -t $SESSION_NAME"
echo ""
echo "2. Detach from session (while inside):"
echo "     Press: Ctrl+B, then D"
echo ""
echo "3. Monitor training logs:"
echo "     tail -f $LOG_FILE"
echo ""
echo "4. Check training progress on WandB:"
echo "     Project: surgical-rgb-conditioned-diffusion"
echo "     Run: $RUN_NAME"
echo ""
echo "5. List all tmux sessions:"
echo "     tmux ls"
echo ""
echo "6. Kill the training session:"
echo "     tmux kill-session -t $SESSION_NAME"
echo ""
echo "========================================================================"
echo "CHECKPOINT LOCATIONS"
echo "========================================================================"
echo ""
echo "Output directory: $OUTPUT_DIR/"
echo "Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "  - best.pt (lowest val loss)"
echo "  - latest.pt (most recent)"
echo "  - epoch_0050.pt, epoch_0100.pt, ... (milestones)"
echo ""
echo "Visualizations: $OUTPUT_DIR/visualizations/"
echo "  - epoch_0050.png, epoch_0100.png, ..."
echo ""
echo "========================================================================"
echo ""
