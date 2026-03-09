"""Token-aware dataset for loading masks with precomputed CLIP tokens.

This dataset loads:
- Coarse and refined masks from disk
- Precomputed RGB CLIP tokens from Step 4
- Applies deterministic preprocessing to masks
- Returns clean samples for diffusion training
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import torch
from torch.utils.data import Dataset
from PIL import Image

from data.transforms import build_transforms, binarize_mask

import cv2
import numpy as np
import yaml
import random


# ============================================================================
# Helper Functions
# ============================================================================

def build_token_path(
    token_dir: Union[str, Path],
    split: str,
    source: str,
    file_stem: str
) -> Path:
    """Construct path to saved token file.
    
    Args:
        token_dir: Base token directory (e.g., 'outputs/clip_tokens')
        split: Split name ('train', 'val', 'test')
        source: Source name ('real_world', 'synthetic')
        file_stem: File stem (e.g., '1672')
        
    Returns:
        Path to token file
        
    Example:
        >>> path = build_token_path('outputs/clip_tokens', 'train', 'real_world', '1672')
        >>> print(path)
        outputs/clip_tokens/train/real_world/1672.pt
    """
    token_dir = Path(token_dir)
    return token_dir / split / source / f"{file_stem}.pt"


def load_token_record(token_path: Union[str, Path]) -> Dict:
    """Load a saved token file and validate basic structure.
    
    Args:
        token_path: Path to token .pt file
        
    Returns:
        Dictionary containing token data and metadata
        
    Raises:
        FileNotFoundError: If token file doesn't exist
        ValueError: If loaded object is not a dict or missing required keys
        
    Example:
        >>> data = load_token_record('outputs/clip_tokens/train/real_world/1672.pt')
        >>> tokens = data['tokens']  # [196, 768]
    """
    token_path = Path(token_path)
    
    if not token_path.exists():
        raise FileNotFoundError(
            f"Token file not found: {token_path}\n"
            f"Have you run Step 4 precomputation for this split/source?"
        )
    
    try:
        data = torch.load(token_path)
    except Exception as e:
        raise ValueError(f"Failed to load token file {token_path}: {e}")
    
    if not isinstance(data, dict):
        raise ValueError(
            f"Token file {token_path} should contain a dict, got {type(data)}"
        )
    
    if 'tokens' not in data:
        raise ValueError(
            f"Token file {token_path} missing required key 'tokens'. "
            f"Available keys: {list(data.keys())}"
        )
    
    return data


def validate_token_record(
    record: Dict,
    expected_source: Optional[str] = None,
    expected_file_stem: Optional[str] = None,
    expected_split: Optional[str] = None
) -> None:
    """Validate token record metadata matches expectations.
    
    Args:
        record: Token record dict from load_token_record()
        expected_source: Expected source name (if provided)
        expected_file_stem: Expected file stem (if provided)
        expected_split: Expected split name (if provided)
        
    Raises:
        ValueError: If validation fails
    """
    # Check token shape
    tokens = record['tokens']
    if not isinstance(tokens, torch.Tensor):
        raise ValueError(f"Expected tokens to be torch.Tensor, got {type(tokens)}")
    
    if tokens.ndim != 2:
        raise ValueError(
            f"Expected tokens to be 2D [num_patches, dim], got shape {tokens.shape}"
        )
    
    # Common expected shape from CLIP ViT-B/16
    if tokens.shape[0] != 196 or tokens.shape[1] != 768:
        raise ValueError(
            f"Expected tokens shape [196, 768], got {tokens.shape}. "
            f"This may indicate mismatched CLIP model or corrupted token file."
        )
    
    # Validate metadata if provided
    if expected_source is not None:
        if 'source' in record and record['source'] != expected_source:
            raise ValueError(
                f"Token record source mismatch: expected '{expected_source}', "
                f"got '{record['source']}'"
            )
    
    if expected_file_stem is not None:
        if 'file_stem' in record and record['file_stem'] != expected_file_stem:
            raise ValueError(
                f"Token record file_stem mismatch: expected '{expected_file_stem}', "
                f"got '{record['file_stem']}'"
            )
    
    if expected_split is not None:
        if 'split' in record and record['split'] != expected_split:
            raise ValueError(
                f"Token record split mismatch: expected '{expected_split}', "
                f"got '{record['split']}'"
            )


# ============================================================================
# Token-Conditioned Mask Dataset
# ============================================================================


class TokenConditionedMaskDataset(Dataset):
    """Dataset that loads masks with precomputed RGB CLIP tokens.
    
    This dataset is designed for training diffusion models that:
    - Take coarse masks as input
    - Generate refined masks as output
    - Use precomputed CLIP RGB tokens as conditioning
    
    The dataset does NOT load RGB images - only masks and tokens.
    This improves efficiency during diffusion training.
    
    Args:
        metadata_dir: Path to metadata directory (e.g., 'data/metadata')
        token_dir: Path to precomputed tokens (e.g., 'outputs/clip_tokens')
        split: Split name ('train', 'val', 'test')
        source: Source filter ('all', 'real_world', 'synthetic')
        image_size: Target image size for masks (default: 512)
        load_spatial_map: If True, load spatial feature maps [768, 14, 14] if available
        return_paths: If True, include file paths in returned sample
        strict_tokens: If True, raise error on missing/invalid tokens (default: True)
        transform: Optional custom transform (if None, uses deterministic eval transform)
        
    Returns:
        Sample dict containing:
        - id: str
        - file_stem: str
        - source: str
        - coarse_mask: Tensor [1, H, W]
        - refined_mask: Tensor [1, H, W]
        - rgb_tokens: Tensor [196, 768]
        - rgb_spatial_map: Tensor [768, 14, 14] (if load_spatial_map=True)
        - coarse_mask_path: str (if return_paths=True)
        - refined_mask_path: str (if return_paths=True)
        - token_path: str (if return_paths=True)
    """
    
    def __init__(
        self,
        metadata_dir: Union[str, Path],
        token_dir: Union[str, Path],
        split: str = 'train',
        source: str = 'all',
        image_size: Union[int, Tuple[int, int]] = 512,
        load_spatial_map: bool = False,
        return_paths: bool = False,
        strict_tokens: bool = True,
        apply_augmentation: bool = True,  # currently always enabled for train split
        transform=None
    ):
        super().__init__()
        
        self.metadata_dir = Path(metadata_dir)
        self.token_dir = Path(token_dir)
        self.split = split
        self.source = source
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.load_spatial_map = load_spatial_map
        self.return_paths = return_paths
        self.strict_tokens = strict_tokens
        
        # Load metadata
        self.samples = self._load_metadata()

        # Augmentation: keep interface unchanged, enable automatically for train split
        self.apply_augmentation = apply_augmentation
        self._load_augmentation_config()
        
        # Build transform
        if transform is None:
            # Keep the same default behavior as the original implementation:
            # deterministic transform pipeline, no transform-level augmentation.
            self.transform = build_transforms(
                image_size=self.image_size[0],  # assumes square for existing pipeline
                train=False,
                augment=False
            )
        else:
            self.transform = transform
        
        print(f"TokenConditionedMaskDataset initialized:")
        print(f"  Split: {split}")
        print(f"  Source: {source}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Image size: {self.image_size}")
        print(f"  Strict tokens: {strict_tokens}")
        print(f"  Apply augmentation: {self.apply_augmentation}")

    def _load_augmentation_config(self) -> None:
        """Load coarse-mask augmentation config from YAML."""
        aug_config_path = Path("configs/train/augmentation.yaml")

        # Defaults match your SurgicalMaskRefinementDataset behavior
        defaults = {
            "augment_prob": 0.5,
            "erode_prob": 0.4,
            "dilate_prob": 0.4,
            "edge_blob_prob": 0.5,
            "drop_parts_prob": 0.4,
            "cutout_prob": 0.01,
            "erode_kernel_range": [3, 9],
            "dilate_kernel_range": [3, 9],
            "erode_iter_range": [1, 2],
            "dilate_iter_range": [1, 2],
            "edge_blob_count_range": [1, 4],
            "edge_blob_radius_range": [4, 16],
            "drop_parts_count_range": [1, 3],
            "drop_parts_radius_range": [6, 18],
            "cutout_count_range": [1, 3],
            "cutout_size_range": [8, 40],
        }

        if aug_config_path.exists():
            with open(aug_config_path, "r") as f:
                aug_config = yaml.safe_load(f) or {}
        else:
            print(f"Warning: augmentation config not found at {aug_config_path}, using defaults.")
            aug_config = {}

        self.augment_prob = aug_config.get("augment_prob", defaults["augment_prob"])
        self.erode_prob = aug_config.get("erode_prob", defaults["erode_prob"])
        self.dilate_prob = aug_config.get("dilate_prob", defaults["dilate_prob"])
        self.edge_blob_prob = aug_config.get("edge_blob_prob", defaults["edge_blob_prob"])
        self.drop_parts_prob = aug_config.get("drop_parts_prob", defaults["drop_parts_prob"])
        self.cutout_prob = aug_config.get("cutout_prob", defaults["cutout_prob"])

        self.erode_kernel_range = tuple(aug_config.get("erode_kernel_range", defaults["erode_kernel_range"]))
        self.dilate_kernel_range = tuple(aug_config.get("dilate_kernel_range", defaults["dilate_kernel_range"]))
        self.erode_iter_range = tuple(aug_config.get("erode_iter_range", defaults["erode_iter_range"]))
        self.dilate_iter_range = tuple(aug_config.get("dilate_iter_range", defaults["dilate_iter_range"]))
        self.edge_blob_count_range = tuple(aug_config.get("edge_blob_count_range", defaults["edge_blob_count_range"]))
        self.edge_blob_radius_range = tuple(aug_config.get("edge_blob_radius_range", defaults["edge_blob_radius_range"]))
        self.drop_parts_count_range = tuple(aug_config.get("drop_parts_count_range", defaults["drop_parts_count_range"]))
        self.drop_parts_radius_range = tuple(aug_config.get("drop_parts_radius_range", defaults["drop_parts_radius_range"]))
        self.cutout_count_range = tuple(aug_config.get("cutout_count_range", defaults["cutout_count_range"]))
        self.cutout_size_range = tuple(aug_config.get("cutout_size_range", defaults["cutout_size_range"]))

        self.seed = aug_config.get("seed", None)

        if self.seed is not None:
            self.rng = random.Random(self.seed)  # fixed seed for reproducibility of augmentations
        else:
            self.rng = random.Random()  # non-deterministic augmentations

    def _load_metadata(self) -> List[Dict]:
        """Load and filter metadata from split file."""
        split_file = self.metadata_dir / f"{self.split}.json"
        
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Available splits: {list(self.metadata_dir.glob('*.json'))}"
            )
        
        with open(split_file, 'r') as f:
            samples = json.load(f)
        
        print(f"Loaded {len(samples)} samples from {split_file}")
        
        # Filter by source
        if self.source != 'all':
            samples = [s for s in samples if s['source'] == self.source]
            print(f"Filtered to {len(samples)} samples (source={self.source})")
        
        if len(samples) == 0:
            raise ValueError(
                f"No samples found for split={self.split}, source={self.source}"
            )
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def _rand_odd(self, low: int, high: int) -> int:
        """Sample an odd integer in [low, high]."""
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
        """Compute a thin band around mask edges."""
        kernel = np.ones((ksize, ksize), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return cv2.subtract(dilated, mask)

    def _add_edge_blobs(self, mask: np.ndarray) -> np.ndarray:
        """Add random blobs along the boundary."""
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
        """Remove random local regions near the boundary."""
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
        """Randomly zero out square regions inside the mask."""
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
        """Apply coarse-mask-only augmentation in numpy space."""
        mask = coarse_mask.copy()

        # print("Wohoo applying augmentation to coarse mask!")

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
        if not self.apply_augmentation:
            return coarse_mask

        if self.rng.random() >= self.augment_prob:
            return coarse_mask
        
        # raise NotImplementedError("Augmentation is currently always enabled for train split. The augment_prob parameter will control the actual augmentation rate. Setting apply_augmentation=False is not implemented yet, but will be the default in the future. Please set apply_augmentation=True for now.")

        coarse_np = self._pil_mask_to_binary_np(coarse_mask)
        coarse_np = self._augment_coarse_mask_only(coarse_np)
        return self._binary_np_to_pil(coarse_np)
    
    def __getitem__(self, idx: int) -> Dict:
        """Load and return a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dict with masks and tokens
        """
        sample_meta = self.samples[idx]
        
        # Extract metadata
        sample_id = sample_meta['id']
        file_stem = sample_meta['file_stem']
        source = sample_meta['source']
        
        # Build paths
        coarse_mask_path = Path(sample_meta['coarse_mask_path'])
        refined_mask_path = Path(sample_meta['refined_mask_path'])
        token_path = build_token_path(self.token_dir, self.split, source, file_stem)
        
        # Load masks
        try:
            coarse_mask = Image.open(coarse_mask_path).convert('L')
            refined_mask = Image.open(refined_mask_path).convert('L')
        except Exception as e:
            raise RuntimeError(
                f"Failed to load masks for sample {sample_id}:\n"
                f"  Coarse: {coarse_mask_path}\n"
                f"  Refined: {refined_mask_path}\n"
                f"  Error: {e}"
            )
        
        # Apply coarse-mask-only augmentation before paired transforms
        try:
            coarse_mask = self._maybe_augment_coarse_mask_pil(coarse_mask)
        except Exception as e:
            raise RuntimeError(
                f"Failed to augment coarse mask for sample {sample_id}:\n"
                f"  Coarse: {coarse_mask_path}\n"
                f"  Error: {e}"
            )

        # Apply transforms (deterministic preprocessing / user-supplied pipeline)
        # Transform expects (rgb, coarse_mask, refined_mask), so we pass dummy RGB.
        try:
            dummy_rgb = Image.new('RGB', coarse_mask.size, color=(0, 0, 0))
            _, coarse_mask_t, refined_mask_t = self.transform(
                dummy_rgb, coarse_mask, refined_mask
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to apply transforms for sample {sample_id}: {e}"
            )
        
        # Load token record
        try:
            token_record = load_token_record(token_path)
            
            if self.strict_tokens:
                validate_token_record(
                    token_record,
                    expected_source=source,
                    expected_file_stem=file_stem,
                    expected_split=self.split
                )
        except (FileNotFoundError, ValueError) as e:
            if self.strict_tokens:
                raise RuntimeError(
                    f"Token validation failed for sample {sample_id}:\n"
                    f"  Token path: {token_path}\n"
                    f"  Error: {e}\n"
                    f"  Hint: Run precomputation for split={self.split}, source={source}"
                ) from e
            else:
                raise NotImplementedError(
                    "Non-strict token loading not fully implemented. "
                    "Please run token precomputation first."
                )
        
        # Extract tokens
        rgb_tokens = token_record['tokens']  # [196, 768]
        
        # Build output sample
        sample = {
            'id': sample_id,
            'file_stem': file_stem,
            'source': source,
            'coarse_mask': coarse_mask_t,      # [1, H, W]
            'refined_mask': refined_mask_t,    # [1, H, W]
            'rgb_tokens': rgb_tokens,          # [196, 768]
        }
        
        # Optional: load spatial map
        if self.load_spatial_map and 'spatial_map' in token_record:
            sample['rgb_spatial_map'] = token_record['spatial_map']  # [768, 14, 14]
        
        # Optional: return paths
        if self.return_paths:
            sample['coarse_mask_path'] = str(coarse_mask_path)
            sample['refined_mask_path'] = str(refined_mask_path)
            sample['token_path'] = str(token_path)
        
        return sample
    
    def get_source_counts(self) -> Dict[str, int]:
        """Return counts of samples by source."""
        from collections import Counter
        return dict(Counter(s['source'] for s in self.samples))
    

# ============================================================================
# Utility Functions
# ============================================================================

def create_token_dataset(
    metadata_dir: str = 'data/metadata',
    token_dir: str = 'outputs/clip_tokens',
    split: str = 'train',
    source: str = 'all',
    **kwargs
) -> TokenConditionedMaskDataset:
    """Convenience factory function for creating token dataset.
    
    Args:
        metadata_dir: Path to metadata directory
        token_dir: Path to precomputed tokens
        split: Split name
        source: Source filter
        **kwargs: Additional arguments for TokenConditionedMaskDataset
        
    Returns:
        TokenConditionedMaskDataset instance
    """
    return TokenConditionedMaskDataset(
        metadata_dir=metadata_dir,
        token_dir=token_dir,
        split=split,
        source=source,
        **kwargs
    )
