#!/usr/bin/env python3
"""Inference and evaluation script for hybrid model on real_world >= 2000."""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.token_dataset import TokenConditionedMaskDataset
from models.diffusion import (
    FrozenVAELatentInterface,
    LatentDiffusionScheduler,
    RGBConditionedLatentDiffusionUNet,
)
from utils.metrics import dice_score, iou_score


def load_yaml(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def clean_state_dict(state_dict: Dict) -> Dict:
    """Remove common prefixes from state dict keys."""
    cleaned = {}
    for k, v in state_dict.items():
        # Remove common prefixes
        for prefix in ('model.', 'module.'):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v
    return cleaned


def sample_ddim(
    model: torch.nn.Module,
    scheduler: LatentDiffusionScheduler,
    z_coarse: torch.Tensor,
    rgb_tokens: torch.Tensor,
    num_inference_steps: int = 50,
    eta: float = 0.0,
) -> torch.Tensor:
    """DDIM sampling with RGB conditioning."""
    device = z_coarse.device
    batch_size = z_coarse.shape[0]
    
    # Start from noise
    z_t = torch.randn_like(z_coarse)
    
    # DDIM timesteps
    timesteps = torch.linspace(
        scheduler.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device
    )
    
    for i, t in enumerate(timesteps):
        t_batch = t.unsqueeze(0).repeat(batch_size)
        
        # Predict noise
        noise_pred = model(z_t, t_batch, z_coarse, rgb_tokens)
        
        # DDIM update
        alpha_t = scheduler.alphas_cumprod[t]
        
        if i < len(timesteps) - 1:
            alpha_t_prev = scheduler.alphas_cumprod[timesteps[i + 1]]
        else:
            alpha_t_prev = torch.tensor(1.0, device=device)
        
        # Predict x0
        pred_x0 = (z_t - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        
        # Direction pointing to z_t
        dir_zt = (1 - alpha_t_prev - eta ** 2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t)).sqrt() * noise_pred
        
        # Random noise
        noise = eta * ((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t)).sqrt() * torch.randn_like(z_t)
        
        # Update
        z_t = alpha_t_prev.sqrt() * pred_x0 + dir_zt + noise
    
    return z_t


def visualize_batch(
    rgb_images: np.ndarray,
    coarse_masks: np.ndarray,
    refined_masks: np.ndarray,
    predictions: np.ndarray,
    dice_scores: List[float],
    iou_scores: List[float],
    sample_ids: List[str],
    output_path: Path,
    max_samples: int = 8
):
    """Create visualization grid for a batch."""
    n_samples = min(len(sample_ids), max_samples)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # RGB image
        axes[i, 0].imshow(rgb_images[i])
        axes[i, 0].set_title(f'RGB\n{sample_ids[i]}')
        axes[i, 0].axis('off')
        
        # Coarse mask overlay
        coarse_overlay = rgb_images[i].copy()
        mask = coarse_masks[i] > 0
        coarse_overlay[mask] = (coarse_overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        axes[i, 1].imshow(coarse_overlay)
        axes[i, 1].set_title('Coarse Mask (Input)')
        axes[i, 1].axis('off')
        
        # Ground truth overlay
        gt_overlay = rgb_images[i].copy()
        mask = refined_masks[i] > 0
        gt_overlay[mask] = (gt_overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        axes[i, 2].imshow(gt_overlay)
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
        
        # Prediction overlay
        pred_overlay = rgb_images[i].copy()
        mask = predictions[i] > 0
        pred_overlay[mask] = (pred_overlay[mask] * 0.5 + np.array([0, 255, 255]) * 0.5).astype(np.uint8)
        axes[i, 3].imshow(pred_overlay)
        axes[i, 3].set_title(f'Prediction\nDice: {dice_scores[i]:.3f} | IoU: {iou_scores[i]:.3f}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/infer/hybrid_inference.yaml')
    parser.add_argument('--split', type=str, default='inference_real_world_ge2000')
    parser.add_argument('--num_vis', type=int, default=24, help='Number of visualization samples')
    args = parser.parse_args()
    
    print("="*70)
    print("HYBRID MODEL INFERENCE ON REAL_WORLD >= 2000")
    print("="*70)
    
    # Load config
    config = load_yaml(args.config)
    device = torch.device(config.get('device', 'cuda'))
    
    # Output directory
    output_dir = Path(config['output']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'predictions').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load VAE
    print("\nLoading VAE...")
    vae_config_path = config['vae_config']
    vae_checkpoint = config['vae_checkpoint']
    vae = FrozenVAELatentInterface(
        model_config_path=vae_config_path,
        checkpoint_path=vae_checkpoint,
        device=device,
        use_mu_only=config.get('vae', {}).get('use_mu_only', True)
    )
    print(f"  ✓ Loaded from {vae_checkpoint}")
    
    # Load diffusion model
    print("\nLoading diffusion model...")
    diffusion_config = load_yaml(config['diffusion_config'])
    model_config = diffusion_config['model']
    rgb_config = diffusion_config['rgb_condition']
    
    model = RGBConditionedLatentDiffusionUNet(
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        base_channels=model_config['base_channels'],
        channel_multipliers=model_config['channel_multipliers'],
        num_res_blocks=model_config['num_res_blocks'],
        time_embed_dim=model_config.get('time_embed_dim', 256),
        norm=model_config.get('norm', 'group'),
        activation=model_config.get('activation', 'silu'),
        dropout=model_config.get('dropout', 0.0),
        rgb_token_dim=rgb_config['token_dim'],
        rgb_projected_dim=rgb_config['projected_dim'],
        rgb_num_heads=rgb_config['num_heads'],
        rgb_dropout=rgb_config.get('dropout', 0.0)
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(config['diffusion_checkpoint'], map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    state_dict = clean_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  ✓ Loaded from {config['diffusion_checkpoint']}")
    if 'epoch' in checkpoint:
        print(f"  ✓ Epoch: {checkpoint['epoch']}")
    
    # Create scheduler
    scheduler = LatentDiffusionScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.02,
        device=str(device)
    )
    
    # Create dataset
    print("\nCreating dataset...")
    data_config = config.get('data', {})
    dataset = TokenConditionedMaskDataset(
        metadata_dir=data_config.get('metadata_dir', 'data/metadata'),
        token_dir=data_config.get('token_dir', 'outputs/clip_tokens'),
        split=args.split,
        source='all',
        image_size=data_config.get('image_size', 512),
        load_spatial_map=False,
        return_paths=True,
        strict_tokens=data_config.get('strict_tokens', True),
        apply_augmentation=False,
        transform=None
    )
    
    print(f"  ✓ {len(dataset)} samples")
    
    # Create dataloader
    batch_size = config.get('batch_size', 16)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Inference settings
    inference_config = config.get('inference', {})
    num_inference_steps = inference_config.get('num_inference_steps', 50)
    eta = inference_config.get('eta', 0.0)
    threshold = config.get('postprocessing', {}).get('threshold', 0.5)
    
    print(f"\nInference settings:")
    print(f"  Batch size: {batch_size}")
    print(f"  DDIM steps: {num_inference_steps}")
    print(f"  Eta: {eta}")
    print(f"  Threshold: {threshold}")
    
    # Run inference
    print("\n" + "="*70)
    print("RUNNING INFERENCE")
    print("="*70)
    
    all_dice = []
    all_iou = []
    saved_vis = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Inference")):
            coarse_mask = batch['coarse_mask'].to(device)
            refined_mask = batch['refined_mask'].to(device)
            rgb_tokens = batch['rgb_tokens'].to(device)
            
            # Encode coarse mask
            z_coarse = vae.encode_coarse_mask(coarse_mask)
            
            # Sample refined latent
            z_pred = sample_ddim(
                model, scheduler, z_coarse, rgb_tokens,
                num_inference_steps=num_inference_steps,
                eta=eta
            )
            
            # Decode to mask
            pred_probs = vae.decode_to_probs(z_pred)
            pred_binary = (pred_probs > threshold).float()
            
            # Calculate metrics
            batch_dice = dice_score(pred_probs, refined_mask, threshold=threshold)
            batch_iou = iou_score(pred_probs, refined_mask, threshold=threshold)
            
            all_dice.extend(batch_dice.cpu().tolist())
            all_iou.extend(batch_iou.cpu().tolist())
            
            # Save predictions
            for i in range(pred_binary.shape[0]):
                sample_id = batch['id'][i]
                pred_np = (pred_binary[i, 0].cpu().numpy() * 255).astype(np.uint8)
                out_path = output_dir / 'predictions' / f'{sample_id}.png'
                Image.fromarray(pred_np).save(out_path)
            
            # Save visualizations
            if saved_vis < args.num_vis:
                num_to_save = min(coarse_mask.shape[0], args.num_vis - saved_vis)
                
                # Load RGB images
                rgb_images = []
                for i in range(num_to_save):
                    refined_path = batch['refined_mask_path'][i]
                    rgb_path = refined_path.replace('refined_mask', 'RGB')
                    rgb_img = np.array(Image.open(rgb_path).convert('RGB').resize((512, 512)))
                    rgb_images.append(rgb_img)
                rgb_images = np.array(rgb_images)
                
                coarse_np = (coarse_mask[:num_to_save, 0].cpu().numpy() * 255).astype(np.uint8)
                refined_np = (refined_mask[:num_to_save, 0].cpu().numpy() * 255).astype(np.uint8)
                pred_np = (pred_binary[:num_to_save, 0].cpu().numpy() * 255).astype(np.uint8)
                
                sample_ids = [batch['id'][i] for i in range(num_to_save)]
                dice_vals = batch_dice[:num_to_save].cpu().tolist()
                iou_vals = batch_iou[:num_to_save].cpu().tolist()
                
                vis_path = output_dir / 'visualizations' / f'batch_{batch_idx:04d}.png'
                visualize_batch(
                    rgb_images, coarse_np, refined_np, pred_np,
                    dice_vals, iou_vals, sample_ids, vis_path,
                    max_samples=num_to_save
                )
                
                saved_vis += num_to_save
    
    # Calculate statistics
    mean_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)
    mean_iou = np.mean(all_iou)
    std_iou = np.std(all_iou)
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Total samples: {len(all_dice)}")
    print(f"\nDice Score:")
    print(f"  Mean: {mean_dice:.4f}")
    print(f"  Std:  {std_dice:.4f}")
    print(f"  Min:  {min(all_dice):.4f}")
    print(f"  Max:  {max(all_dice):.4f}")
    print(f"\nIoU Score:")
    print(f"  Mean: {mean_iou:.4f}")
    print(f"  Std:  {std_iou:.4f}")
    print(f"  Min:  {min(all_iou):.4f}")
    print(f"  Max:  {max(all_iou):.4f}")
    
    # Save results
    results = {
        'num_samples': len(all_dice),
        'dice': {
            'mean': float(mean_dice),
            'std': float(std_dice),
            'min': float(min(all_dice)),
            'max': float(max(all_dice)),
            'per_sample': all_dice
        },
        'iou': {
            'mean': float(mean_iou),
            'std': float(std_iou),
            'min': float(min(all_iou)),
            'max': float(max(all_iou)),
            'per_sample': all_iou
        },
        'config': {
            'checkpoint': config['diffusion_checkpoint'],
            'split': args.split,
            'num_inference_steps': num_inference_steps,
            'threshold': threshold
        }
    }
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    print(f"✓ Predictions saved to {output_dir / 'predictions'}")
    print(f"✓ Visualizations saved to {output_dir / 'visualizations'}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
