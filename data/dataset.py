"""PyTorch dataset classes for surgical mask data.

Provides dataset classes for loading paired RGB images and masks
with optional augmentation.
"""

from pathlib import Path
from typing import Dict, Optional, Callable, List, Union
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import cv2
import yaml

class SurgicalMaskRefinementDataset(Dataset):
    """Unified dataset for paired RGB images and segmentation masks.
    
    Loads RGB, coarse_mask, and refined_mask triplets from split files.
    Supports flexible loading modes for different use cases.
    
    Args:
        metadata_dir: Directory containing split JSON files
        split: Which split to load ('train', 'val', 'test')
        source: Which source to load ('all', 'real_world', 'synthetic')
        load_images: Whether to load actual image data (default: True)
        return_paths: Whether to include file paths in returned samples (default: False)
        apply_transforms: Whether to apply preprocessing transforms (default: False)
        transform: Optional paired transform callable (if None, no transforms applied)
        
    Example:
        >>> # Load training data with transforms
        >>> from data.transforms import build_transforms
        >>> transform = build_transforms(train=True, augment=True)
        >>> dataset = SurgicalMaskRefinementDataset(
        ...     metadata_dir='data/metadata',
        ...     split='train',
        ...     apply_transforms=True,
        ...     transform=transform
        ... )
        >>> sample = dataset[0]
        >>> print(sample['rgb'].shape)  # torch.Size([3, 512, 512])
        
        >>> # Load validation data, paths only (for precomputing)
        >>> dataset = SurgicalMaskRefinementDataset(
        ...     metadata_dir='data/metadata',
        ...     split='val',
        ...     load_images=False,
        ...     return_paths=True
        ... )
        >>> sample = dataset[0]
        >>> print(sample['rgb_path'])  # Path to RGB image
    """
    
    def __init__(
        self,
        metadata_dir: Union[str, Path],
        split: str = "train",
        source: str = "all",
        load_images: bool = True,
        return_paths: bool = False,
        apply_transforms: bool = False,
        transform: Optional[Callable] = None,
        apply_augmentation: bool = True,
    ):
        """Initialize dataset."""
        self.metadata_dir = Path(metadata_dir)
        self.split = split
        self.source = source
        self.load_images = load_images
        self.return_paths = return_paths
        self.apply_transforms = apply_transforms
        self.transform = transform

        self.apply_augmentation = apply_augmentation
        
        # self.augment_prob = 0.5

        # self.erode_prob = 0.4
        # self.dilate_prob = 0.4
        # self.edge_blob_prob = 0.5
        # self.drop_parts_prob = 0.4
        # self.cutout_prob = 0.01

        # self.erode_kernel_range = (3, 9)
        # self.dilate_kernel_range = (3, 9)
        # self.erode_iter_range = (1, 2)
        # self.dilate_iter_range = (1, 2)

        # self.edge_blob_count_range = (1, 4)
        # self.edge_blob_radius_range = (4, 16)
        # self.drop_parts_count_range = (1, 3)
        # self.drop_parts_radius_range = (6, 18)
        # self.cutout_count_range = (1, 3)
        # self.cutout_size_range = (8, 40)

        # Read from ./config/train/augmentation.yaml
        with open("configs/train/augmentation.yaml", "r") as f:
            aug_config = yaml.safe_load(f)
        self.augment_prob = aug_config.get("augment_prob", 0.5)
        self.erode_prob = aug_config.get("erode_prob", 0.4)
        self.dilate_prob = aug_config.get("dilate_prob", 0.4)
        self.edge_blob_prob = aug_config.get("edge_blob_prob", 0.5)
        self.drop_parts_prob = aug_config.get("drop_parts_prob", 0.4)
        self.cutout_prob = aug_config.get("cutout_prob", 0.01)
        self.erode_kernel_range = tuple(aug_config.get("erode_kernel_range", [3, 9]))
        self.dilate_kernel_range = tuple(aug_config.get("dilate_kernel_range", [3, 9]))
        self.erode_iter_range = tuple(aug_config.get("erode_iter_range", [1, 2]))
        self.dilate_iter_range = tuple(aug_config.get("dilate_iter_range", [1, 2]))
        self.edge_blob_count_range = tuple(aug_config.get("edge_blob_count_range", [1, 4]))
        self.edge_blob_radius_range = tuple(aug_config.get("edge_blob_radius_range", [4, 16]))
        self.drop_parts_count_range = tuple(aug_config.get("drop_parts_count_range", [1, 3]))
        self.drop_parts_radius_range = tuple(aug_config.get("drop_parts_radius_range", [6, 18]))
        self.cutout_count_range = tuple(aug_config.get("cutout_count_range", [1, 3]))
        

        self.rng = random.Random()
        
        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'")
        
        # Load samples from split file
        split_file = self.metadata_dir / f"{split}.json"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Have you run the split generation script?"
            )
        
        with open(split_file, 'r') as f:
            all_samples = json.load(f)
        
        # Filter by source if requested
        if source == "all":
            self.samples = all_samples
        elif source in ["real_world", "synthetic"]:
            self.samples = [s for s in all_samples if s['source'] == source]
        else:
            raise ValueError(
                f"Invalid source '{source}'. Must be 'all', 'real_world', or 'synthetic'"
            )
        
        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found for split '{split}' and source '{source}'"
            )
        
        print(f"Loaded {len(self.samples)} samples from {split} split (source: {source})")
        
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def _load_image(self, path: str) -> Image.Image:
        """Load an image from disk.
        
        Args:
            path: Path to image file
            
        Returns:
            PIL Image
        """
        return Image.open(path)
    
    def _rand_odd(self, low: int, high: int) -> int:
        k = self.rng.randint(low, high)
        if k % 2 == 0:
            k += 1
        return k

    def _pil_mask_to_binary_np(self, mask: Image.Image) -> np.ndarray:
        """Convert PIL mask to binary uint8 numpy array with values {0,255}."""
        arr = np.array(mask.convert("L"), dtype=np.uint8)
        _, arr = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
        return arr

    def _binary_np_to_pil(self, mask: np.ndarray) -> Image.Image:
        """Convert binary uint8 numpy mask back to PIL Image."""
        mask = np.ascontiguousarray(mask.astype(np.uint8))
        return Image.fromarray(mask, mode="L")

    def _edge_band(self, mask: np.ndarray, ksize: int = 5) -> np.ndarray:
        kernel = np.ones((ksize, ksize), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return cv2.subtract(dilated, mask)

    def _add_edge_blobs(self, mask: np.ndarray) -> np.ndarray:
        edge = self._edge_band(mask, ksize=5)
        ys, xs = np.where(edge > 0)

        if len(xs) == 0:
            return mask

        out = mask.copy()
        n_blobs = self.rng.randint(*self.edge_blob_count_range)

        for _ in range(n_blobs):
            idx = self.rng.randrange(len(xs))
            x, y = int(xs[idx]), int(ys[idx])
            radius = self.rng.randint(*self.edge_blob_radius_range)
            cv2.circle(out, (x, y), radius, 255, thickness=-1)

        return out

    def _drop_parts(self, mask: np.ndarray) -> np.ndarray:
        edge = self._edge_band(mask, ksize=3)
        ys, xs = np.where(edge > 0)

        if len(xs) == 0:
            return mask

        out = mask.copy()
        n_parts = self.rng.randint(*self.drop_parts_count_range)

        for _ in range(n_parts):
            idx = self.rng.randrange(len(xs))
            x, y = int(xs[idx]), int(ys[idx])
            radius = self.rng.randint(*self.drop_parts_radius_range)
            cv2.circle(out, (x, y), radius, 0, thickness=-1)

        return out

    def _random_cutout(self, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            return mask

        out = mask.copy()
        n_cutouts = self.rng.randint(*self.cutout_count_range)

        for _ in range(n_cutouts):
            idx = self.rng.randrange(len(xs))
            cx, cy = int(xs[idx]), int(ys[idx])
            half = self.rng.randint(*self.cutout_size_range)

            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(mask.shape[1], cx + half)
            y2 = min(mask.shape[0], cy + half)

            out[y1:y2, x1:x2] = 0

        return out

    def _augment_coarse_mask_only(self, coarse_mask: np.ndarray) -> np.ndarray:
        mask = coarse_mask.copy()

        if self.rng.random() < self.erode_prob:
            k = self._rand_odd(*self.erode_kernel_range)
            iters = self.rng.randint(*self.erode_iter_range)
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=iters)

        if self.rng.random() < self.dilate_prob:
            k = self._rand_odd(*self.dilate_kernel_range)
            iters = self.rng.randint(*self.dilate_iter_range)
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=iters)

        if self.rng.random() < self.edge_blob_prob:
            mask = self._add_edge_blobs(mask)

        if self.rng.random() < self.drop_parts_prob:
            mask = self._drop_parts(mask)

        if self.rng.random() < self.cutout_prob:
            mask = self._random_cutout(mask)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def _maybe_augment_coarse_mask_pil(self, coarse_mask: Image.Image) -> Image.Image:
        """Apply coarse-mask-only augmentation, preserving PIL interface."""

        if self.rng.random() >= self.augment_prob:
            return coarse_mask

        coarse_np = self._pil_mask_to_binary_np(coarse_mask)
        coarse_np = self._augment_coarse_mask_only(coarse_np)
        return self._binary_np_to_pil(coarse_np)
        
    def __getitem__(self, idx: int) -> Dict[str, Union[str, Image.Image, torch.Tensor]]:
        """Load and return a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample data based on configuration:
                - 'id': Unique sample identifier (always included)
                - 'file_stem': Original filename stem (always included)
                - 'source': Source type (always included)
                - 'rgb': RGB PIL Image or tensor (if load_images=True)
                - 'coarse_mask': Coarse mask PIL Image or tensor (if load_images=True)
                - 'refined_mask': Refined mask PIL Image or tensor (if load_images=True)
                - 'rgb_path': Path to RGB image (if return_paths=True)
                - 'coarse_mask_path': Path to coarse mask (if return_paths=True)
                - 'refined_mask_path': Path to refined mask (if return_paths=True)
        """
        sample_meta = self.samples[idx]
        
        # Always include metadata
        sample = {
            'id': sample_meta['id'],
            'file_stem': sample_meta['file_stem'],
            'source': sample_meta['source'],
        }
        
        # Add paths if requested
        if self.return_paths:
            sample['rgb_path'] = sample_meta['rgb_path']
            sample['coarse_mask_path'] = sample_meta['coarse_mask_path']
            sample['refined_mask_path'] = sample_meta['refined_mask_path']
        
        # Load images if requested
        if self.load_images:
            # Load RGB image
            rgb = self._load_image(sample_meta['rgb_path'])
            
            # Load masks
            coarse_mask = self._load_image(sample_meta['coarse_mask_path'])
            refined_mask = self._load_image(sample_meta['refined_mask_path'])

            if self.apply_augmentation:
                coarse_mask = self._maybe_augment_coarse_mask_pil(coarse_mask) # Only augment the coarse mask
            
            # Apply paired transforms if requested
            if self.apply_transforms and self.transform is not None:
                rgb, coarse_mask, refined_mask = self.transform(rgb, coarse_mask, refined_mask)
            
            sample['rgb'] = rgb
            sample['coarse_mask'] = coarse_mask
            sample['refined_mask'] = refined_mask
        
        return sample
    
    def get_source_counts(self) -> Dict[str, int]:
        """Get counts of samples by source.
        
        Returns:
            Dictionary mapping source name to count
        """
        from collections import defaultdict
        counts = defaultdict(int)
        for sample in self.samples:
            counts[sample['source']] += 1
        return dict(counts)


class VAEDataset(Dataset):
    """Dataset for VAE training on masks only.
    
    Loads only mask images for VAE pretraining.
    This is a convenience wrapper around SurgicalMaskRefinementDataset
    that focuses on masks.
    
    Args:
        metadata_dir: Directory containing split JSON files
        split: Which split to load ('train', 'val', 'test')
        source: Which source to load ('all', 'real_world', 'synthetic')
        mask_type: Which mask to load ('refined', 'coarse', 'both')
        apply_transforms: Whether to apply transforms (default: False)
        mask_transform: Optional transform for masks (should handle single mask)
    """
    
    def __init__(
        self,
        metadata_dir: Union[str, Path],
        split: str = "train",
        source: str = "all",
        mask_type: str = "refined",
        apply_transforms: bool = False,
        mask_transform: Optional[Callable] = None,
    ):
        """Initialize VAE dataset."""
        self.base_dataset = SurgicalMaskRefinementDataset(
            metadata_dir=metadata_dir,
            split=split,
            source=source,
            load_images=True,
            return_paths=False,
            apply_transforms=False,  # We handle transforms separately for masks
            transform=None,
        )
        
        if mask_type not in ['refined', 'coarse', 'both']:
            raise ValueError(
                f"Invalid mask_type '{mask_type}'. Must be 'refined', 'coarse', or 'both'"
            )
        
        self.mask_type = mask_type
        self.apply_transforms = apply_transforms
        self.mask_transform = mask_transform
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, Image.Image, torch.Tensor]]:
        """Load and return a mask sample.
        
        Returns:
            Dictionary containing:
                - 'id': Sample identifier
                - 'source': Source type
                - 'mask': Mask image/tensor (if mask_type is 'refined' or 'coarse')
                - 'refined_mask': Refined mask (if mask_type is 'both')
                - 'coarse_mask': Coarse mask (if mask_type is 'both')
        """
        sample = self.base_dataset[idx]
        
        result = {
            'id': sample['id'],
            'source': sample['source'],
        }
        
        if self.mask_type == 'refined':
            mask = sample['refined_mask']
            if self.apply_transforms and self.mask_transform is not None:
                mask = self.mask_transform(mask)
            result['mask'] = mask
        elif self.mask_type == 'coarse':
            mask = sample['coarse_mask']
            if self.apply_transforms and self.mask_transform is not None:
                mask = self.mask_transform(mask)
            result['mask'] = mask
        else:  # both
            refined = sample['refined_mask']
            coarse = sample['coarse_mask']
            if self.apply_transforms and self.mask_transform is not None:
                refined = self.mask_transform(refined)
                coarse = self.mask_transform(coarse)
            result['refined_mask'] = refined
            result['coarse_mask'] = coarse
        
        return result
