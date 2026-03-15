"""
Conditional VAE (CVAE) for Mask Refinement
Takes coarse_mask and RGB as input, outputs refined mask
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime
from models.baselines.cvae import *

# Import augmented dataset if available
try:
    from data.baselien_aug_dataset import SegRefineDataset
    AUG_DATASET_AVAILABLE = True
except ImportError:
    AUG_DATASET_AVAILABLE = False


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice coefficient between prediction and target
    Args:
        pred: Predicted mask (tensor)
        target: Ground truth mask (tensor)
        threshold: Threshold for binarization
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Dice coefficient (float)
    """
    # Binarize predictions
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    
    # Dice = 2 * |A ∩ B| / (|A| + |B|)
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice.item()


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union) between prediction and target
    Args:
        pred: Predicted mask (tensor)
        target: Ground truth mask (tensor)
        threshold: Threshold for binarization
        smooth: Smoothing factor to avoid division by zero
    Returns:
        IoU score (float)
    """
    # Binarize predictions
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    # IoU = |A ∩ B| / |A ∪ B|
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


class MaskRefinementDataset(Dataset):
    """Dataset for mask refinement task"""
    
    def __init__(self, dataset_path, dataset_type='real_world', target_size=(256, 256)):
        """
        Args:
            dataset_path: Path to dataset root
            dataset_type: 'real_world', 'synthetic', or 'both'
            target_size: Resize images to this size (H, W)
        """
        self.target_size = target_size
        self.data_list = []
        
        # Load real_world dataset
        if dataset_type in ['real_world', 'both']:
            real_path = os.path.join(dataset_path, 'real_world')
            rgb_files = sorted(os.listdir(os.path.join(real_path, 'RGB')))
            for fname in rgb_files:
                base_name = fname
                rgb_path = os.path.join(real_path, 'RGB', base_name)
                coarse_path = os.path.join(real_path, 'coarse_mask', base_name)
                refined_path = os.path.join(real_path, 'refined_mask', base_name)
                
                if os.path.exists(coarse_path) and os.path.exists(refined_path):
                    self.data_list.append({
                        'rgb': rgb_path,
                        'coarse': coarse_path,
                        'refined': refined_path,
                        'source': 'real_world'
                    })
        
        # Load synthetic dataset
        if dataset_type in ['synthetic', 'both']:
            syn_path = os.path.join(dataset_path, 'synthetic')
            rgb_files = sorted(os.listdir(os.path.join(syn_path, 'RGB')))
            for fname in rgb_files:
                base_name = fname
                rgb_path = os.path.join(syn_path, 'RGB', base_name)
                coarse_path = os.path.join(syn_path, 'coarse_mask', base_name)
                refined_path = os.path.join(syn_path, 'refined_mask', base_name)
                
                if os.path.exists(coarse_path) and os.path.exists(refined_path):
                    self.data_list.append({
                        'rgb': rgb_path,
                        'coarse': coarse_path,
                        'refined': refined_path,
                        'source': 'synthetic'
                    })
        
        print(f"Loaded {len(self.data_list)} samples for {dataset_type} dataset")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        # Load images
        rgb = Image.open(data['rgb']).convert('RGB')
        coarse = Image.open(data['coarse']).convert('L')  # Grayscale for mask
        refined = Image.open(data['refined']).convert('L')
        
        # Apply transforms
        rgb = self.transform(rgb)
        coarse = self.transform(coarse)
        refined = self.transform(refined)
        
        # Concatenate RGB and coarse mask as condition
        condition = torch.cat([rgb, coarse], dim=0)  # 4 channels
        
        return condition, refined, data['source']


class AugmentedDatasetAdapter(Dataset):
    """Adapter to make SegRefineDataset compatible with training pipeline"""
    
    def __init__(self, dataset_path, target_size=(256, 256), seed=42):
        """
        Args:
            dataset_path: Path to dataset root
            target_size: Resize images to this size (H, W)
            seed: Random seed for augmentation
        """
        if not AUG_DATASET_AVAILABLE:
            raise ImportError("aug_dataset.py not found. Cannot use augmented dataset.")
        
        # Initialize the augmented dataset with both real_world and synthetic
        self.aug_dataset = SegRefineDataset(
            root=dataset_path,
            domains=('real_world', 'synthetic'),
            target_size=target_size,
            augment_prob=0.8,  # High probability for augmentation
            seed=seed
        )
        
        # Normalization for converting uint8 to float [0, 1]
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.aug_dataset)
    
    def __getitem__(self, idx):
        # Get sample from augmented dataset
        sample = self.aug_dataset[idx]
        
        # Extract components
        rgb = sample['rgb']  # HxWx3 uint8 numpy array
        refined_mask = sample['refined_mask']  # HxW uint8 numpy array
        aug_coarse_mask = sample['aug_coarse_mask']  # HxW uint8 numpy array (augmented)
        source = sample['domain']  # 'real_world' or 'synthetic'
        
        # Convert to PIL Images for transform pipeline
        from PIL import Image
        rgb_pil = Image.fromarray(rgb)
        refined_pil = Image.fromarray(refined_mask)
        aug_coarse_pil = Image.fromarray(aug_coarse_mask)
        
        # Convert to tensors [0, 1]
        rgb_tensor = self.to_tensor(rgb_pil)  # 3xHxW
        refined_tensor = self.to_tensor(refined_pil)  # 1xHxW
        aug_coarse_tensor = self.to_tensor(aug_coarse_pil)  # 1xHxW
        
        # Concatenate RGB and augmented coarse mask as condition
        condition = torch.cat([rgb_tensor, aug_coarse_tensor], dim=0)  # 4 channels
        
        return condition, refined_tensor, source


class Sim2RealDatasetAdapter(Dataset):
    """Adapter for sim2real: trains on augmented synthetic, evaluates on real-world"""
    
    def __init__(self, dataset_path, mode='train', target_size=(256, 256), seed=42):
        """
        Args:
            dataset_path: Path to dataset root
            mode: 'train' (synthetic with augmentation) or 'val' (real-world without augmentation)
            target_size: Resize images to this size (H, W)
            seed: Random seed for augmentation
        """
        self.mode = mode
        
        if mode == 'train':
            # Training: use synthetic data with heavy augmentation
            if not AUG_DATASET_AVAILABLE:
                raise ImportError("aug_dataset.py not found. Cannot use sim2real mode.")
            
            self.aug_dataset = SegRefineDataset(
                root=dataset_path,
                domains=('synthetic',),  # Only synthetic for training
                target_size=target_size,
                augment_prob=0.9,  # Very high augmentation for sim2real
                seed=seed
            )
            
            self.to_tensor = transforms.Compose([transforms.ToTensor()])
            
        else:  # mode == 'val'
            # Validation: use real-world data WITHOUT augmentation
            self.real_dataset = MaskRefinementDataset(
                dataset_path=dataset_path,
                dataset_type='real_world',
                target_size=target_size
            )
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.aug_dataset)
        else:
            return len(self.real_dataset)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            # Get augmented synthetic sample
            sample = self.aug_dataset[idx]
            
            rgb = sample['rgb']
            refined_mask = sample['refined_mask']
            aug_coarse_mask = sample['aug_coarse_mask']
            source = 'synthetic'  # Always synthetic in training
            
            # Convert to tensors
            from PIL import Image
            rgb_tensor = self.to_tensor(Image.fromarray(rgb))
            refined_tensor = self.to_tensor(Image.fromarray(refined_mask))
            aug_coarse_tensor = self.to_tensor(Image.fromarray(aug_coarse_mask))
            
            condition = torch.cat([rgb_tensor, aug_coarse_tensor], dim=0)
            return condition, refined_tensor, source
            
        else:  # val mode
            # Get real-world sample without augmentation
            return self.real_dataset[idx]


def evaluate_model(model, dataloader, device, num_samples=5):
    """Evaluate model and compute metrics"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for condition, target, _ in dataloader:
            condition = condition.to(device)
            target = target.to(device)
            
            recon, mu, logvar = model(condition, target)
            loss, recon_loss, kl_loss = cvae_loss(recon, target, mu, logvar)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon = total_recon_loss / len(dataloader.dataset)
    avg_kl = total_kl_loss / len(dataloader.dataset)
    
    return avg_loss, avg_recon, avg_kl


def save_generated_samples(model, dataloader, device, save_path, num_samples=4, threshold=0.5, dataset_type='real_world'):
    """Save generated samples for visualization with Dice and IoU metrics"""
    model.eval()
    
    # Collect samples
    conditions = []
    targets = []
    sources = []
    
    # Gather samples from dataloader
    with torch.no_grad():
        for condition, target, source in dataloader:
            for i in range(len(condition)):
                conditions.append(condition[i])
                targets.append(target[i])
                sources.append(source[i])
            
            if len(conditions) >= num_samples * 3:  # Get extra samples for selection
                break
    
    # If dataset_type is 'both' or 'both_augmented', try to get balanced samples from both sources
    selected_indices = []
    if dataset_type in ['both', 'both_augmented']:
        real_indices = [i for i, s in enumerate(sources) if s == 'real_world']
        syn_indices = [i for i, s in enumerate(sources) if s == 'synthetic']
        
        # Try to get equal samples from each source
        samples_per_source = num_samples // 2
        selected_indices.extend(real_indices[:samples_per_source])
        selected_indices.extend(syn_indices[:samples_per_source])
        
        # Fill remaining slots
        remaining = num_samples - len(selected_indices)
        all_indices = list(set(range(len(conditions))) - set(selected_indices))
        selected_indices.extend(all_indices[:remaining])
    else:
        selected_indices = list(range(min(num_samples, len(conditions))))
    
    # Stack selected samples
    selected_conditions = torch.stack([conditions[i] for i in selected_indices]).to(device)
    selected_targets = torch.stack([targets[i] for i in selected_indices]).to(device)
    selected_sources = [sources[i] for i in selected_indices]
    
    with torch.no_grad():
        recon, _, _ = model(selected_conditions, selected_targets)
    
    # Create visualization
    n_samples = len(selected_indices)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Calculate metrics for this sample
        dice = dice_coefficient(recon[i], selected_targets[i], threshold)
        iou = iou_score(recon[i], selected_targets[i], threshold)
        
        # RGB
        rgb = selected_conditions[i, :3].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(rgb)
        # Special label for sim2real mode
        if dataset_type == 'sim2real':
            source_label = 'Real (val)'
        else:
            source_label = 'Real' if selected_sources[i] == 'real_world' else 'Syn'
        axes[i, 0].set_title(f'RGB Input ({source_label})')
        axes[i, 0].axis('off')
        
        # Coarse mask
        coarse = selected_conditions[i, 3].cpu().numpy()
        axes[i, 1].imshow(coarse, cmap='gray')
        axes[i, 1].set_title('Coarse Mask')
        axes[i, 1].axis('off')
        
        # Ground truth
        gt = selected_targets[i, 0].cpu().numpy()
        axes[i, 2].imshow(gt, cmap='gray')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
        
        # Generated with metrics in title
        gen = recon[i, 0].cpu().numpy()
        axes[i, 3].imshow(gen, cmap='gray')
        axes[i, 3].set_title(f'Generated\nDice: {dice:.4f} | IoU: {iou:.4f}')
        axes[i, 3].axis('off')
    
    # Add subtitle for sim2real mode
    if dataset_type == 'sim2real':
        fig.suptitle('Sim2Real: Trained on Synthetic, Evaluated on Real-World', fontsize=14, y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def final_evaluation(model, dataloader, device, output_dir, dataset_type, threshold=0.5, num_vis_samples=10):
    """Evaluate model on entire validation set and save detailed results"""
    print("\n" + "="*60)
    print("Final Evaluation on Validation Set")
    print("="*60)
    
    model.eval()
    all_dice = []
    all_iou = []
    
    # Collect samples for visualization
    vis_conditions = []
    vis_targets = []
    vis_preds = []
    vis_sources = []
    
    with torch.no_grad():
        for batch_idx, (condition, target, source) in enumerate(tqdm(dataloader, desc='Evaluating')):
            condition = condition.to(device)
            target = target.to(device)
            
            # Generate predictions
            pred, _, _ = model(condition, target)
            
            # Calculate metrics for each sample in batch
            for i in range(condition.size(0)):
                dice = dice_coefficient(pred[i], target[i], threshold)
                iou = iou_score(pred[i], target[i], threshold)
                
                all_dice.append(dice)
                all_iou.append(iou)
                
                # Collect samples for visualization - intelligently select for 'both' dataset
                if dataset_type in ['both', 'both_augmented']:
                    # Try to balance real_world and synthetic samples
                    real_count = sum(1 for s in vis_sources if s == 'real_world')
                    syn_count = sum(1 for s in vis_sources if s == 'synthetic')
                    
                    if len(vis_conditions) < num_vis_samples:
                        # Add sample if we need more of this type
                        if source[i] == 'real_world' and (real_count < num_vis_samples // 2 or syn_count >= num_vis_samples // 2):
                            vis_conditions.append(condition[i].cpu())
                            vis_targets.append(target[i].cpu())
                            vis_preds.append(pred[i].cpu())
                            vis_sources.append(source[i])
                        elif source[i] == 'synthetic' and (syn_count < num_vis_samples // 2 or real_count >= num_vis_samples // 2):
                            vis_conditions.append(condition[i].cpu())
                            vis_targets.append(target[i].cpu())
                            vis_preds.append(pred[i].cpu())
                            vis_sources.append(source[i])
                else:
                    if len(vis_conditions) < num_vis_samples:
                        vis_conditions.append(condition[i].cpu())
                        vis_targets.append(target[i].cpu())
                        vis_preds.append(pred[i].cpu())
                        vis_sources.append(source[i])
    
    # Calculate statistics
    mean_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)
    mean_iou = np.mean(all_iou)
    std_iou = np.std(all_iou)
    
    # Print results
    print(f"\nEvaluation Results on {len(all_dice)} samples:")
    print(f"  Dice Coefficient: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"  IoU Score:        {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"  Min Dice: {min(all_dice):.4f} | Max Dice: {max(all_dice):.4f}")
    print(f"  Min IoU:  {min(all_iou):.4f} | Max IoU:  {max(all_iou):.4f}")
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'final_evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Final Evaluation Results - {dataset_type}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total samples evaluated: {len(all_dice)}\n\n")
        f.write(f"Dice Coefficient:\n")
        f.write(f"  Mean: {mean_dice:.6f}\n")
        f.write(f"  Std:  {std_dice:.6f}\n")
        f.write(f"  Min:  {min(all_dice):.6f}\n")
        f.write(f"  Max:  {max(all_dice):.6f}\n\n")
        f.write(f"IoU Score:\n")
        f.write(f"  Mean: {mean_iou:.6f}\n")
        f.write(f"  Std:  {std_iou:.6f}\n")
        f.write(f"  Min:  {min(all_iou):.6f}\n")
        f.write(f"  Max:  {max(all_iou):.6f}\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create visualization with metrics
    num_vis = min(len(vis_conditions), num_vis_samples)
    fig, axes = plt.subplots(num_vis, 4, figsize=(16, 4*num_vis))
    if num_vis == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_vis):
        dice = dice_coefficient(vis_preds[i], vis_targets[i], threshold)
        iou = iou_score(vis_preds[i], vis_targets[i], threshold)
        
        # RGB
        rgb = vis_conditions[i][:3].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(rgb)
        # Special label for sim2real mode
        if dataset_type == 'sim2real':
            source_label = 'Real (val)'
        else:
            source_label = 'Real' if vis_sources[i] == 'real_world' else 'Syn'
        axes[i, 0].set_title(f'RGB Input ({source_label})')
        axes[i, 0].axis('off')
        
        # Coarse mask
        coarse = vis_conditions[i][3].numpy()
        axes[i, 1].imshow(coarse, cmap='gray')
        axes[i, 1].set_title('Coarse Mask')
        axes[i, 1].axis('off')
        
        # Ground truth
        gt = vis_targets[i][0].numpy()
        axes[i, 2].imshow(gt, cmap='gray')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
        
        # Generated
        gen = vis_preds[i][0].numpy()
        axes[i, 3].imshow(gen, cmap='gray')
        axes[i, 3].set_title(f'Generated\nDice: {dice:.4f} | IoU: {iou:.4f}')
        axes[i, 3].axis('off')
    
    # Add subtitle for sim2real mode
    if dataset_type == 'sim2real':
        fig.suptitle(f'Sim2Real Final Evaluation - Trained on Synthetic, Tested on Real-World\nMean Dice: {mean_dice:.4f} | Mean IoU: {mean_iou:.4f}', 
                     fontsize=14, y=0.998)
    else:
        plt.suptitle(f'Final Evaluation - Mean Dice: {mean_dice:.4f} | Mean IoU: {mean_iou:.4f}', 
                     fontsize=16, y=0.995)
    plt.tight_layout()
    
    vis_path = os.path.join(output_dir, 'final_evaluation_samples.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {vis_path}")
    print("="*60 + "\n")
    
    return mean_dice, mean_iou, std_dice, std_iou


def plot_loss_curves(train_losses, val_losses, save_path):
    """Plot and save loss curves"""
    epochs = range(1, len(train_losses['total']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Total loss
    axes[0].plot(epochs, train_losses['total'], 'b-', label='Train')
    axes[0].plot(epochs, val_losses['total'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Reconstruction loss
    axes[1].plot(epochs, train_losses['recon'], 'b-', label='Train')
    axes[1].plot(epochs, val_losses['recon'], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # KL divergence
    axes[2].plot(epochs, train_losses['kl'], 'b-', label='Train')
    axes[2].plot(epochs, val_losses['kl'], 'r-', label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"{args.dataset_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Create datasets
    if args.dataset_type == 'both_augmented':
        if not AUG_DATASET_AVAILABLE:
            raise RuntimeError("aug_dataset.py not found. Cannot use both_augmented mode.")
        print("Using augmented dataset mode (real_world + synthetic with augmentation)")
        full_dataset = AugmentedDatasetAdapter(
            args.dataset_path,
            target_size=(args.image_size, args.image_size),
            seed=42
        )
        # Split for training
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
    elif args.dataset_type == 'sim2real':
        if not AUG_DATASET_AVAILABLE:
            raise RuntimeError("aug_dataset.py not found. Cannot use sim2real mode.")
        print("Using sim2real mode: Training on augmented synthetic, evaluating on real-world")
        # Training: augmented synthetic data
        train_dataset = Sim2RealDatasetAdapter(
            args.dataset_path,
            mode='train',
            target_size=(args.image_size, args.image_size),
            seed=42
        )
        # Validation: real-world data (no augmentation)
        val_dataset = Sim2RealDatasetAdapter(
            args.dataset_path,
            mode='val',
            target_size=(args.image_size, args.image_size),
            seed=42
        )
        print(f"Train samples: {len(train_dataset)} (synthetic + aug), Validation samples: {len(val_dataset)} (real-world)")
        
    else:
        full_dataset = MaskRefinementDataset(
            args.dataset_path, 
            args.dataset_type,
            target_size=(args.image_size, args.image_size)
        )
        # Split for training
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    if args.dataset_type != 'sim2real':
        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = CVAE(condition_channels=4, latent_dim=args.latent_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    train_losses = {'total': [], 'recon': [], 'kl': []}
    val_losses = {'total': [], 'recon': [], 'kl': []}
    
    best_val_loss = float('inf')
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        # Training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for condition, target, _ in pbar:
            condition = condition.to(device)
            target = target.to(device)
            
            # Forward pass
            recon, mu, logvar = model(condition, target)
            
            # Compute loss
            loss, recon_loss, kl_loss = cvae_loss(
                recon, target, mu, logvar, 
                kl_weight=args.kl_weight
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item() / len(condition),
                'recon': recon_loss.item() / len(condition),
                'kl': kl_loss.item() / len(condition)
            })
        
        # Average training losses
        avg_train_loss = epoch_loss / len(train_dataset)
        avg_train_recon = epoch_recon / len(train_dataset)
        avg_train_kl = epoch_kl / len(train_dataset)
        
        train_losses['total'].append(avg_train_loss)
        train_losses['recon'].append(avg_train_recon)
        train_losses['kl'].append(avg_train_kl)
        
        # Validation
        avg_val_loss, avg_val_recon, avg_val_kl = evaluate_model(model, val_loader, device)
        
        val_losses['total'].append(avg_val_loss)
        val_losses['recon'].append(avg_val_recon)
        val_losses['kl'].append(avg_val_kl)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save checkpoint every n epochs
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            print(f"  New best model saved! Val loss: {best_val_loss:.4f}")
        
        # Generate and save samples
        if epoch % args.eval_every == 0:
            sample_path = os.path.join(output_dir, 'samples', f'samples_epoch_{epoch}.png')
            save_generated_samples(model, val_loader, device, sample_path, 
                                 num_samples=5, threshold=args.threshold, dataset_type=args.dataset_type)
            print(f"  Saved samples: {sample_path}")
        
        # Plot loss curves
        plot_path = os.path.join(output_dir, 'plots', 'loss_curves.png')
        plot_loss_curves(train_losses, val_losses, plot_path)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {output_dir}")
    
    # Final evaluation on validation set with best model
    print("\nLoading best model for final evaluation...")
    best_model_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    mean_dice, mean_iou, std_dice, std_iou = final_evaluation(
        model, val_loader, device, output_dir, args.dataset_type, 
        threshold=args.threshold, num_vis_samples=10
    )
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Dice Score:     {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Final IoU Score:      {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Results Directory:    {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train CVAE for Mask Refinement')
    
    # Data parameters
    parser.add_argument('--dataset_path', type=str, 
                        default='/home/kuancheng/Desktop/285/ece285_dataset',
                        help='Path to dataset')
    parser.add_argument('--dataset_type', type=str, default='real_world',
                        choices=['real_world', 'synthetic', 'both', 'both_augmented', 'sim2real'],
                        help='Which dataset to use (both_augmented uses aug_dataset.py, sim2real trains on synthetic and evaluates on real_world)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--kl_weight', type=float, default=0.1,
                        help='Weight for KL divergence loss')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Saving parameters
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every n epochs')
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Evaluate and save samples every n epochs')
    
    # Evaluation parameters
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binarizing masks when computing metrics')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*50)
    print("CVAE Training Configuration")
    print("="*50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("="*50 + "\n")
    
    # Train
    train(args)


if __name__ == '__main__':
    main()
