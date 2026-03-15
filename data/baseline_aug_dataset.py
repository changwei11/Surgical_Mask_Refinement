import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


@dataclass
class SampleRecord:
    domain: str
    name: str
    rgb_path: str
    refined_mask_path: str
    coarse_mask_path: str


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SegRefineDataset(Dataset):
    def __init__(
        self,
        root: str = "data",
        domains: Tuple[str, ...] = ("real_world", "synthetic"),
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
        target_size: Optional[Tuple[int, int]] = (700, 493),
        augment_prob: float = 0.5,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        erode_prob: float = 0.4,
        dilate_prob: float = 0.4,
        edge_blob_prob: float = 0.5,
        drop_parts_prob: float = 0.4,
        cutout_prob: float = 0.05,
        erode_kernel_range: Tuple[int, int] = (3, 9),
        dilate_kernel_range: Tuple[int, int] = (3, 9),
        erode_iter_range: Tuple[int, int] = (1, 2),
        dilate_iter_range: Tuple[int, int] = (1, 2),
        edge_blob_count_range: Tuple[int, int] = (1, 4),
        edge_blob_radius_range: Tuple[int, int] = (4, 16),
        drop_parts_count_range: Tuple[int, int] = (1, 3),
        drop_parts_radius_range: Tuple[int, int] = (6, 18),
        cutout_count_range: Tuple[int, int] = (1, 3),
        cutout_size_range: Tuple[int, int] = (8, 40),
        seed: Optional[int] = None,
    ):
        self.root = root
        self.domains = domains
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)

        self.target_size = target_size

        self.augment_prob = augment_prob

        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

        self.erode_prob = erode_prob
        self.dilate_prob = dilate_prob
        self.edge_blob_prob = edge_blob_prob
        self.drop_parts_prob = drop_parts_prob
        self.cutout_prob = cutout_prob

        self.erode_kernel_range = erode_kernel_range
        self.dilate_kernel_range = dilate_kernel_range
        self.erode_iter_range = erode_iter_range
        self.dilate_iter_range = dilate_iter_range

        self.edge_blob_count_range = edge_blob_count_range
        self.edge_blob_radius_range = edge_blob_radius_range
        self.drop_parts_count_range = drop_parts_count_range
        self.drop_parts_radius_range = drop_parts_radius_range
        self.cutout_count_range = cutout_count_range
        self.cutout_size_range = cutout_size_range

        self.rng = random.Random(seed)
        self.samples: List[SampleRecord] = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid triplets found under: {root}")

    @classmethod
    def from_yaml(cls, yaml_path: str):
        cfg = load_yaml_config(yaml_path)
        return cls(
            root=cfg["root"],
            domains=tuple(cfg["domains"]),
            image_extensions=tuple(cfg["image_extensions"]),
            target_size=tuple(cfg["target_size"]),
            augment_prob=cfg["augment_prob"],
            hflip_prob=cfg["hflip_prob"],
            vflip_prob=cfg["vflip_prob"],
            erode_prob=cfg["erode_prob"],
            dilate_prob=cfg["dilate_prob"],
            edge_blob_prob=cfg["edge_blob_prob"],
            drop_parts_prob=cfg["drop_parts_prob"],
            cutout_prob=cfg["cutout_prob"],
            erode_kernel_range=tuple(cfg["erode_kernel_range"]),
            dilate_kernel_range=tuple(cfg["dilate_kernel_range"]),
            erode_iter_range=tuple(cfg["erode_iter_range"]),
            dilate_iter_range=tuple(cfg["dilate_iter_range"]),
            edge_blob_count_range=tuple(cfg["edge_blob_count_range"]),
            edge_blob_radius_range=tuple(cfg["edge_blob_radius_range"]),
            drop_parts_count_range=tuple(cfg["drop_parts_count_range"]),
            drop_parts_radius_range=tuple(cfg["drop_parts_radius_range"]),
            cutout_count_range=tuple(cfg["cutout_count_range"]),
            cutout_size_range=tuple(cfg["cutout_size_range"]),
            seed=cfg["seed"],
        )

    def _collect_samples(self) -> List[SampleRecord]:
        samples: List[SampleRecord] = []

        for domain in self.domains:
            rgb_dir = os.path.join(self.root, domain, "RGB")
            refined_dir = os.path.join(self.root, domain, "refined_mask")
            coarse_dir = os.path.join(self.root, domain, "coarse_mask")

            if not (os.path.isdir(rgb_dir) and os.path.isdir(refined_dir) and os.path.isdir(coarse_dir)):
                continue

            for name in sorted(os.listdir(rgb_dir)):
                if not name.lower().endswith(self.image_extensions):
                    continue

                rgb_path = os.path.join(rgb_dir, name)
                refined_path = os.path.join(refined_dir, name)
                coarse_path = os.path.join(coarse_dir, name)

                if not (os.path.isfile(refined_path) and os.path.isfile(coarse_path)):
                    continue

                samples.append(
                    SampleRecord(
                        domain=domain,
                        name=name,
                        rgb_path=rgb_path,
                        refined_mask_path=refined_path,
                        coarse_mask_path=coarse_path,
                    )
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _rand_odd(self, low: int, high: int) -> int:
        k = self.rng.randint(low, high)
        if k % 2 == 0:
            k += 1
        return k

    def _load_mask(self, path: str, shape_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")

        if shape_hw is not None and mask.shape != shape_hw:
            mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

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

    def __getitem__(self, idx: int):
        record = self.samples[idx]

        rgb = cv2.imread(record.rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read RGB image: {record.rgb_path}")
        
        # Convert BGR to RGB (OpenCV loads in BGR format)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        refined_mask = self._load_mask(record.refined_mask_path)
        coarse_mask = self._load_mask(record.coarse_mask_path, shape_hw=refined_mask.shape)

        target_w, target_h = self.target_size

        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        refined_mask = cv2.resize(refined_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        coarse_mask = cv2.resize(coarse_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        if self.rng.random() < self.hflip_prob:
            rgb = cv2.flip(rgb, 1)
            refined_mask = cv2.flip(refined_mask, 1)
            coarse_mask = cv2.flip(coarse_mask, 1)

        if self.rng.random() < self.vflip_prob:
            rgb = cv2.flip(rgb, 0)
            refined_mask = cv2.flip(refined_mask, 0)
            coarse_mask = cv2.flip(coarse_mask, 0)

        aug_coarse_mask = coarse_mask.copy()
        if self.rng.random() < self.augment_prob:
            aug_coarse_mask = self._augment_coarse_mask_only(coarse_mask)

        rgb = np.ascontiguousarray(rgb)
        refined_mask = np.ascontiguousarray(refined_mask)
        coarse_mask = np.ascontiguousarray(coarse_mask)
        aug_coarse_mask = np.ascontiguousarray(aug_coarse_mask)

        return {
            "domain": record.domain,
            "name": record.name,
            "rgb": rgb,
            "refined_mask": refined_mask,
            "coarse_mask": coarse_mask,
            "aug_coarse_mask": aug_coarse_mask,
        }


def mask_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.copy()
    color_img = np.zeros_like(image)
    color_img[:] = color
    mask_bool = mask > 0
    out[mask_bool] = cv2.addWeighted(image[mask_bool], 1 - alpha, color_img[mask_bool], alpha, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def make_viewer_canvas(sample: dict) -> np.ndarray:
    rgb = sample["rgb"]
    refined = sample["refined_mask"]
    coarse = sample["coarse_mask"]
    aug = sample["aug_coarse_mask"]

    overlay_refined = mask_overlay(rgb, refined, (0, 255, 0))
    overlay_coarse = mask_overlay(rgb, coarse, (0, 255, 255))
    overlay_aug = mask_overlay(rgb, aug, (255, 0, 255))
    
    canvas = np.hstack([overlay_refined, overlay_coarse, overlay_aug])

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    labels = [
        ("Refined Mask", (10, 30)),
        ("Coarse Mask", (rgb.shape[1] + 10, 30)),
        ("Aug Coarse Mask", (2 * rgb.shape[1] + 10, 30)),
    ]
    
    for text, pos in labels:
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(canvas, (pos[0] - 5, pos[1] - text_height - 5), 
                      (pos[0] + text_width + 5, pos[1] + baseline + 5), bg_color, -1)
        cv2.putText(canvas, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

    return canvas


if __name__ == "__main__":
    dataset = SegRefineDataset.from_yaml("config/train_set.yaml")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    cv2.namedWindow("Sample Viewer", cv2.WINDOW_NORMAL)

    for batch in dataloader:
        print(f"Batch loaded: {batch['name']}")
        print(f"RGB shape: {batch['rgb'].shape}")
        print(f"Refined mask shape: {batch['refined_mask'].shape}")

        sample = {
            "domain": batch["domain"][0],
            "name": batch["name"][0],
            "rgb": batch["rgb"][0].numpy().astype(np.uint8),
            "refined_mask": batch["refined_mask"][0].numpy().astype(np.uint8),
            "coarse_mask": batch["coarse_mask"][0].numpy().astype(np.uint8),
            "aug_coarse_mask": batch["aug_coarse_mask"][0].numpy().astype(np.uint8),
        }

        canvas = make_viewer_canvas(sample)
        cv2.imshow("Sample Viewer", canvas)

        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()
