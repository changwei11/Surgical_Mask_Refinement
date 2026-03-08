"""Training script for RGB-conditioned latent diffusion.

Entry point for RGB-conditioned latent diffusion training with CLIP tokens.
"""

import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import sys
import logging
import random
import numpy as np
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import MaskVAE
from models.diffusion import (
    FrozenVAELatentInterface,
    RGBConditionedLatentDiffusionUNet,
    LatentDiffusionScheduler,
    DiffusionLoss
)
from data.token_dataset import TokenConditionedMaskDataset
from trainers.rgb_diffusion_trainer import RGBConditionedLatentDiffusionTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RGB-conditioned latent diffusion with CLIP tokens")
    
    # Config files
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train/diffusion_rgb_train.yaml",
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--diffusion_config",
        type=str,
        default="configs/model/diffusion_rgb.yaml",
        help="Path to diffusion model config YAML"
    )
    parser.add_argument(
        "--vae_config",
        type=str,
        default="configs/model/vae.yaml",
        help="Path to VAE model config YAML"
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="outputs/vae/checkpoints/best.pt",
        help="Path to trained VAE checkpoint"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on"
    )
    
    # Overfit mode for debugging
    parser.add_argument(
        "--overfit_small",
        action="store_true",
        help="Enable overfit mode on small subset for debugging"
    )
    parser.add_argument(
        "--overfit_num_samples",
        type=int,
        default=16,
        help="Number of samples for overfit mode"
    )
    
    # Training overrides
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--eval_every_n_epochs",
        type=int,
        default=None,
        help="Override evaluation frequency"
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help="Override checkpoint save frequency"
    )
    
    # Checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    
    # WandB
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="WandB run name"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable WandB logging"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_frozen_vae_interface(
    vae_config_path: str,
    vae_checkpoint_path: str,
    use_mu_only: bool,
    device: str
) -> FrozenVAELatentInterface:
    """Create frozen VAE interface."""
    # Check checkpoint exists
    if not Path(vae_checkpoint_path).exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint_path}")
    
    # Create interface
    vae_interface = FrozenVAELatentInterface(
        model_config_path=vae_config_path,
        checkpoint_path=vae_checkpoint_path,
        device=device,
        use_mu_only=use_mu_only
    )
    
    logger.info(f"Frozen VAE interface created:")
    logger.info(f"  Checkpoint: {vae_checkpoint_path}")
    logger.info(f"  Frozen: {vae_interface.is_frozen()}")
    logger.info(f"  Total parameters: {vae_interface.count_total_parameters():,}")
    logger.info(f"  Trainable parameters: {vae_interface.count_trainable_parameters()}")
    logger.info(f"  Latent shape: {vae_interface.latent_shape}")
    logger.info(f"  Use mu only: {use_mu_only}")
    
    return vae_interface


def create_diffusion_model(diffusion_config: dict, device: str) -> RGBConditionedLatentDiffusionUNet:
    """Create RGB-conditioned diffusion U-Net model from config."""
    model_config = diffusion_config['model']
    rgb_config = diffusion_config.get('rgb_condition', {})
    
    model = RGBConditionedLatentDiffusionUNet(
        # Base U-Net params
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        base_channels=model_config['base_channels'],
        channel_multipliers=model_config['channel_multipliers'],
        num_res_blocks=model_config['num_res_blocks'],
        time_embed_dim=model_config['time_embed_dim'],
        norm=model_config.get('norm', 'group'),
        activation=model_config.get('activation', 'silu'),
        dropout=model_config.get('dropout', 0.0),
        # RGB conditioning params
        rgb_token_dim=rgb_config.get('token_dim', 768),
        rgb_projected_dim=rgb_config.get('projected_dim', 256),
        rgb_num_heads=rgb_config.get('num_heads', 4),
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"RGB-Conditioned Diffusion U-Net created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  RGB token dim: {rgb_config.get('token_dim', 768)}")
    logger.info(f"  RGB projected dim: {rgb_config.get('projected_dim', 256)}")
    logger.info(f"  RGB attention heads: {rgb_config.get('num_heads', 4)}")
    
    return model.to(device)


def create_scheduler(diffusion_config: dict, device: str) -> LatentDiffusionScheduler:
    """Create diffusion scheduler from config."""
    scheduler_config = diffusion_config['scheduler']
    
    scheduler = LatentDiffusionScheduler(
        num_train_timesteps=scheduler_config['num_train_timesteps'],
        beta_schedule=scheduler_config['beta_schedule'],
        beta_start=scheduler_config['beta_start'],
        beta_end=scheduler_config['beta_end'],
        device=device
    )
    
    logger.info(f"Diffusion scheduler created:")
    logger.info(f"  Timesteps: {scheduler_config['num_train_timesteps']}")
    logger.info(f"  Beta schedule: {scheduler_config['beta_schedule']}")
    logger.info(f"  Beta range: [{scheduler_config['beta_start']}, {scheduler_config['beta_end']}]")
    
    return scheduler


def create_datasets(train_config: dict, overfit_mode: bool = False, overfit_num: int = 16):
    """Create token-conditioned train and validation datasets."""
    data_config = train_config['data']
    
    # Create datasets with precomputed tokens
    train_dataset = TokenConditionedMaskDataset(
        metadata_dir=data_config['metadata_dir'],
        token_dir=data_config['token_dir'],
        split=data_config['split_train'],
        source=data_config['source'],
        image_size=data_config['image_size'],
        load_spatial_map=False,
        return_paths=False,
        strict_tokens=data_config.get('strict_tokens', True),
        transform=None  # Uses deterministic transforms by default
    )
    
    val_dataset = TokenConditionedMaskDataset(
        metadata_dir=data_config['metadata_dir'],
        token_dir=data_config['token_dir'],
        split=data_config['split_val'],
        source=data_config['source'],
        image_size=data_config['image_size'],
        load_spatial_map=False,
        return_paths=False,
        strict_tokens=data_config.get('strict_tokens', True),
        transform=None  # Uses deterministic transforms by default
    )

    # assert train_dataset.apply_augmentation == True and val_dataset.apply_augmentation == True, "Datasets should apply augmentations by default"
    
    # Overfit mode: use small subset
    if overfit_mode:
        logger.info(f"OVERFIT MODE: Using only {overfit_num} samples")
        train_indices = list(range(min(overfit_num, len(train_dataset))))
        val_indices = list(range(min(max(overfit_num // 2, 1), len(val_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    logger.info(f"Token-conditioned datasets created:")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int,
    overfit_mode: bool = False
) -> tuple:
    """Create dataloaders."""
    # For overfit mode with very small dataset, don't drop last batch
    drop_last_train = not overfit_mode and len(train_dataset) > batch_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last_train
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_optimizer(model: torch.nn.Module, train_config: dict) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    opt_config = train_config.get('optimizer', {})
    opt_type = opt_config.get('type', 'adamw').lower()
    
    lr = train_config['train']['learning_rate']
    weight_decay = train_config['train'].get('weight_decay', 0.0)
    
    if opt_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=opt_config.get('betas', (0.9, 0.999)),
            eps=opt_config.get('eps', 1e-8)
        )
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=opt_config.get('betas', (0.9, 0.999)),
            eps=opt_config.get('eps', 1e-8)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    logger.info(f"Optimizer: {opt_type}, LR: {lr}, Weight decay: {weight_decay}")
    
    return optimizer


def create_lr_scheduler(optimizer: torch.optim.Optimizer, train_config: dict):
    """Create learning rate scheduler if configured."""
    scheduler_config = train_config.get('scheduler', {})
    
    if not scheduler_config.get('use_scheduler', False):
        return None
    
    scheduler_type = scheduler_config.get('type', 'cosine').lower()
    epochs = train_config['train']['epochs']
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=scheduler_config.get('eta_min', 0)
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, not using scheduler")
        return None
    
    logger.info(f"LR Scheduler: {scheduler_type}")
    return scheduler


def init_wandb(train_config: dict, diffusion_config: dict, vae_config: dict, run_name: str = None):
    """Initialize Weights & Biases."""
    wandb_config = train_config.get('wandb', {})
    
    if not wandb_config.get('use_wandb', True):
        logger.info("WandB disabled by config")
        return False
    
    # Check if wandb is available
    try:
        import wandb
    except ImportError:
        logger.error("WandB not installed. Install with: pip install wandb")
        raise
    
    # Initialize wandb
    project = wandb_config.get('project', 'surgical-rgb-conditioned-diffusion')
    name = run_name or wandb_config.get('run_name', None)
    mode = wandb_config.get('mode', 'online')
    tags = wandb_config.get('tags', [])
    
    wandb.init(
        project=project,
        name=name,
        config={
            'train': train_config['train'],
            'data': train_config['data'],
            'diffusion_model': diffusion_config['model'],
            'diffusion_scheduler': diffusion_config['scheduler'],
            'rgb_condition': diffusion_config.get('rgb_condition', {}),
            'vae': train_config['vae'],
        },
        mode=mode,
        tags=tags
    )
    
    logger.info(f"WandB initialized: project={project}, name={name}, mode={mode}")
    return True


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 70)
    print("RGB-CONDITIONED LATENT DIFFUSION TRAINING (WITH CLIP TOKENS)")
    print("=" * 70)
    
    # Load configs
    logger.info(f"Loading training config from {args.train_config}")
    train_config = load_config(args.train_config)
    
    logger.info(f"Loading diffusion config from {args.diffusion_config}")
    diffusion_config = load_config(args.diffusion_config)
    
    logger.info(f"Loading VAE config from {args.vae_config}")
    vae_config = load_config(args.vae_config)
    
    # Apply CLI overrides
    if args.epochs is not None:
        train_config['train']['epochs'] = args.epochs
        logger.info(f"Override epochs: {args.epochs}")
    
    if args.batch_size is not None:
        train_config['train']['batch_size'] = args.batch_size
        logger.info(f"Override batch_size: {args.batch_size}")
    
    if args.learning_rate is not None:
        train_config['train']['learning_rate'] = args.learning_rate
        logger.info(f"Override learning_rate: {args.learning_rate}")
    
    if args.eval_every_n_epochs is not None:
        train_config['eval']['eval_every_n_epochs'] = args.eval_every_n_epochs
        logger.info(f"Override eval_every_n_epochs: {args.eval_every_n_epochs}")
    
    if args.save_every_n_epochs is not None:
        train_config['checkpoint']['save_every_n_epochs'] = args.save_every_n_epochs
        logger.info(f"Override save_every_n_epochs: {args.save_every_n_epochs}")
    
    if args.output_dir is not None:
        train_config['checkpoint']['output_dir'] = args.output_dir
        logger.info(f"Override output_dir: {args.output_dir}")
    
    if args.no_wandb:
        train_config['wandb']['use_wandb'] = False
        logger.info("WandB disabled by CLI")
    
    # Set seed
    seed = train_config['train'].get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    logger.info(f"Device: {device}")
    
    # Initialize WandB
    use_wandb = init_wandb(train_config, diffusion_config, vae_config, args.run_name) if not args.no_wandb else False
    
    # Create frozen VAE interface
    logger.info("Creating frozen VAE interface...")
    vae_interface = create_frozen_vae_interface(
        vae_config_path=args.vae_config,
        vae_checkpoint_path=args.vae_checkpoint,
        use_mu_only=train_config['vae'].get('use_mu_only', True),
        device=device
    )
    
    # Create RGB-conditioned diffusion model
    logger.info("Creating RGB-conditioned diffusion model...")
    diffusion_model = create_diffusion_model(diffusion_config, device)
    
    # Create scheduler
    logger.info("Creating diffusion scheduler...")
    scheduler = create_scheduler(diffusion_config, device)
    
    # Create datasets
    logger.info("Creating token-conditioned datasets...")
    train_dataset, val_dataset = create_datasets(
        train_config,
        overfit_mode=args.overfit_small,
        overfit_num=args.overfit_num_samples
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=train_config['train']['batch_size'],
        num_workers=train_config['train']['num_workers'],
        overfit_mode=args.overfit_small
    )
    
    # Create criterion
    logger.info("Creating loss function...")
    criterion = DiffusionLoss(
        loss_type=diffusion_config['loss']['type'],
        reduction=diffusion_config['loss']['reduction']
    )
    logger.info(f"  Loss type: {diffusion_config['loss']['type']}")
    logger.info(f"  Reduction: {diffusion_config['loss']['reduction']}")
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(diffusion_model, train_config)
    
    # Create LR scheduler
    lr_scheduler = create_lr_scheduler(optimizer, train_config)
    
    # Create trainer
    logger.info("Creating RGB-conditioned trainer...")
    trainer = RGBConditionedLatentDiffusionTrainer(
        model=diffusion_model,
        vae_interface=vae_interface,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=train_config,
        device=device,
        lr_scheduler=lr_scheduler,
        use_wandb=use_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint_wrapper(args.resume)
    
    # Start training
    logger.info("\n" + "=" * 70)
    logger.info("STARTING RGB-CONDITIONED TRAINING")
    logger.info("=" * 70)
    
    try:
        trainer.fit()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint_wrapper('interrupted.pt')
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        if use_wandb:
            wandb.finish()
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    main()
