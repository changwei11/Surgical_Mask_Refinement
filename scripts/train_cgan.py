"""
Conditional GAN
"""

import os
import sys

# Add project root to path so that 'models', 'data', etc. are importable
# regardless of which directory the script is run from.
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
from models.baselines.cgan import *

# Import augmented dataset if available
try:
    from data.baseline_aug_dataset import SegRefineDataset
    AUG_DATASET_AVAILABLE = True
except ImportError:
    AUG_DATASET_AVAILABLE = False


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    """
    Dice coefficient (float)
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice.item()


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    IoU score (float)
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
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


class Generator(nn.Module):
    """U-Net style Generator for mask refinement"""
    
    def __init__(self, input_channels=4, output_channels=1, ngf=64):
        super(Generator, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(input_channels, ngf, normalize=False)  # 256x256
        self.enc2 = self.conv_block(ngf, ngf * 2)  # 128x128
        self.enc3 = self.conv_block(ngf * 2, ngf * 4)  # 64x64
        self.enc4 = self.conv_block(ngf * 4, ngf * 8)  # 32x32
        self.enc5 = self.conv_block(ngf * 8, ngf * 8)  # 16x16
        self.enc6 = self.conv_block(ngf * 8, ngf * 8)  # 8x8
        
        # Decoder (upsampling)
        self.dec1 = self.deconv_block(ngf * 8, ngf * 8, dropout=True)  # 16x16
        self.dec2 = self.deconv_block(ngf * 16, ngf * 8, dropout=True)  # 32x32
        self.dec3 = self.deconv_block(ngf * 16, ngf * 4)  # 64x64
        self.dec4 = self.deconv_block(ngf * 8, ngf * 2)  # 128x128
        self.dec5 = self.deconv_block(ngf * 4, ngf)  # 256x256
        
        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_channels, 4, 2, 1),
            nn.Sigmoid()
        )
        
    def conv_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def deconv_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        
        # Decoder with skip connections
        d1 = self.dec1(e6)
        d2 = self.dec2(torch.cat([d1, e5], dim=1))
        d3 = self.dec3(torch.cat([d2, e4], dim=1))
        d4 = self.dec4(torch.cat([d3, e3], dim=1))
        d5 = self.dec5(torch.cat([d4, e2], dim=1))
        
        # Final output
        out = self.final(torch.cat([d5, e1], dim=1))
        
        return out


class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    
    def __init__(self, input_channels=5, ndf=64):
        super(Discriminator, self).__init__()
        
        # input is (condition + refined_mask) = 5 channels
        self.model = nn.Sequential(
            # 256x256
            nn.Conv2d(input_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 31x31
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
            # Output: 30x30 patch
        )
    
    def forward(self, condition, mask):
        x = torch.cat([condition, mask], dim=1)
        return self.model(x)


def evaluate_model(generator, dataloader, device):
    """Evaluate generator and compute average loss"""
    generator.eval()
    total_l1_loss = 0
    
    with torch.no_grad():
        for condition, target, _ in dataloader:
            condition = condition.to(device)
            target = target.to(device)
            
            # Generate
            fake = generator(condition)
            
            # L1 loss
            l1_loss = F.l1_loss(fake, target, reduction='sum')
            total_l1_loss += l1_loss.item()
    
    avg_l1_loss = total_l1_loss / len(dataloader.dataset)
    return avg_l1_loss


def save_generated_samples(generator, dataloader, device, save_path, num_samples=4, threshold=0.5, dataset_type='real_world'):
    """Save generated samples for visualization with Dice and IoU metrics"""
    generator.eval()
    
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
    
    # If dataset_type is 'both', try to get balanced samples from both sources
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
        generated = generator(selected_conditions)
    
    # Create visualization
    n_samples = len(selected_indices)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Calculate metrics for this sample
        dice = dice_coefficient(generated[i], selected_targets[i], threshold)
        iou = iou_score(generated[i], selected_targets[i], threshold)
        
        # RGB
        rgb = selected_conditions[i, :3].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(rgb)
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
        gen = generated[i, 0].cpu().numpy()
        axes[i, 3].imshow(gen, cmap='gray')
        axes[i, 3].set_title(f'Generated\nDice: {dice:.4f} | IoU: {iou:.4f}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def final_evaluation(generator, dataloader, device, output_dir, dataset_type, threshold=0.5, num_vis_samples=10):
    """Evaluate model on entire validation set and save detailed results"""
    print("\n" + "="*60)
    print("Final Evaluation on Validation Set")
    print("="*60)
    
    generator.eval()
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
            pred = generator(condition)
            
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
    epochs = range(1, len(train_losses['g_total']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Generator Total Loss
    axes[0, 0].plot(epochs, train_losses['g_total'], 'b-', label='Train')
    axes[0, 0].plot(epochs, val_losses['l1'], 'r-', label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Generator Loss')
    axes[0, 0].set_title('Generator Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Generator Adversarial Loss
    axes[0, 1].plot(epochs, train_losses['g_adv'], 'b-', label='Train')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Adversarial Loss')
    axes[0, 1].set_title('Generator Adversarial Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Generator L1 Loss
    axes[1, 0].plot(epochs, train_losses['g_l1'], 'b-', label='Train')
    axes[1, 0].plot(epochs, val_losses['l1'], 'r-', label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L1 Loss')
    axes[1, 0].set_title('Generator L1 Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Discriminator Loss
    axes[1, 1].plot(epochs, train_losses['d_loss'], 'b-', label='Train')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Discriminator Loss')
    axes[1, 1].set_title('Discriminator Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
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
    output_dir = os.path.join(args.output_dir, f"cgan_{args.dataset_type}_{timestamp}")
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
    
    # Create models
    generator = Generator(input_channels=4, output_channels=1, ngf=args.ngf).to(device)
    discriminator = Discriminator(input_channels=5, ndf=args.ndf).to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters())}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters())}")
    
    # Loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    
    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', factor=0.5, patience=5
    )
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    train_losses = {'g_total': [], 'g_adv': [], 'g_l1': [], 'd_loss': []}
    val_losses = {'l1': []}
    
    best_val_loss = float('inf')
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        
        epoch_g_total = 0
        epoch_g_adv = 0
        epoch_g_l1 = 0
        epoch_d_loss = 0
        
        # Training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for condition, target, _ in pbar:
            batch_size = condition.size(0)
            condition = condition.to(device)
            target = target.to(device)
            
            # Real and fake labels
            real_label = torch.ones(batch_size, 1, 30, 30, device=device)
            fake_label = torch.zeros(batch_size, 1, 30, 30, device=device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()
            
            # Real loss
            real_pred = discriminator(condition, target)
            loss_d_real = criterion_gan(real_pred, real_label)
            
            # Fake loss
            fake = generator(condition)
            fake_pred = discriminator(condition, fake.detach())
            loss_d_fake = criterion_gan(fake_pred, fake_label)
            
            # Total discriminator loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()
            
            # -----------------
            # Train Generator
            # -----------------
            optimizer_g.zero_grad()
            
            # Adversarial loss
            fake_pred = discriminator(condition, fake)
            loss_g_adv = criterion_gan(fake_pred, real_label)
            
            # L1 loss
            loss_g_l1 = criterion_l1(fake, target)
            
            # Total generator loss
            loss_g = loss_g_adv + args.lambda_l1 * loss_g_l1
            loss_g.backward()
            optimizer_g.step()
            
            # Update metrics
            epoch_g_total += loss_g.item()
            epoch_g_adv += loss_g_adv.item()
            epoch_g_l1 += loss_g_l1.item()
            epoch_d_loss += loss_d.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G': f'{loss_g.item():.4f}',
                'D': f'{loss_d.item():.4f}',
                'L1': f'{loss_g_l1.item():.4f}'
            })
        
        # Average training losses
        num_batches = len(train_loader)
        avg_g_total = epoch_g_total / num_batches
        avg_g_adv = epoch_g_adv / num_batches
        avg_g_l1 = epoch_g_l1 / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        
        train_losses['g_total'].append(avg_g_total)
        train_losses['g_adv'].append(avg_g_adv)
        train_losses['g_l1'].append(avg_g_l1)
        train_losses['d_loss'].append(avg_d_loss)
        
        # Validation
        avg_val_l1 = evaluate_model(generator, val_loader, device)
        val_losses['l1'].append(avg_val_l1)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train - G_total: {avg_g_total:.4f}, G_adv: {avg_g_adv:.4f}, G_L1: {avg_g_l1:.4f}, D: {avg_d_loss:.4f}")
        print(f"  Val   - L1: {avg_val_l1:.4f}")
        
        # Learning rate scheduling
        scheduler_g.step(avg_val_l1)
        scheduler_d.step(avg_d_loss)
        
        # Save checkpoint every n epochs
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_val_l1 < best_val_loss:
            best_val_loss = avg_val_l1
            best_model_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            print(f"  New best model saved! Val L1: {best_val_loss:.4f}")
        
        # Generate and save samples
        if epoch % args.eval_every == 0:
            sample_path = os.path.join(output_dir, 'samples', f'samples_epoch_{epoch}.png')
            save_generated_samples(generator, val_loader, device, sample_path, 
                                 num_samples=5, threshold=args.threshold, dataset_type=args.dataset_type)
            print(f"  Saved samples: {sample_path}")
        
        # Plot loss curves
        plot_path = os.path.join(output_dir, 'plots', 'loss_curves.png')
        plot_loss_curves(train_losses, val_losses, plot_path)
    
    print("\nTraining completed!")
    print(f"Best validation L1 loss: {best_val_loss:.4f}")
    print(f"Results saved to: {output_dir}")
    
    # Final evaluation on validation set with best model
    print("\nLoading best model for final evaluation...")
    best_model_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
    checkpoint = torch.load(best_model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    mean_dice, mean_iou, std_dice, std_iou = final_evaluation(
        generator, val_loader, device, output_dir, args.dataset_type, 
        threshold=args.threshold, num_vis_samples=10
    )
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Best Validation L1 Loss: {best_val_loss:.4f}")
    print(f"Final Dice Score:        {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Final IoU Score:         {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Results Directory:       {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train cGAN for Mask Refinement')
    
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
    parser.add_argument('--ngf', type=int, default=64,
                        help='Number of generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64,
                        help='Number of discriminator filters in first conv layer')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--lambda_l1', type=float, default=100.0,
                        help='Weight for L1 loss')
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
    print("cGAN Training Configuration")
    print("="*50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("="*50 + "\n")
    
    # Train
    train(args)


if __name__ == '__main__':
    main()
