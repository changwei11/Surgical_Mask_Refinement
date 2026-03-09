"""Inference script for generating refined masks.

Runs trained diffusion model on test data or specific samples.
Supports optional test-time augmentation and visualization outputs.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import SurgicalMaskRefinementDataset
from data.transforms import build_transforms
from models.diffusion import (
    FrozenVAELatentInterface,
    LatentDiffusionScheduler,
    LatentDiffusionUNet,
)
from utils.metrics import dice_score, iou_score


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run diffusion inference")
    parser.add_argument("--config", type=str, required=True, help="Path to inference config YAML")
    parser.add_argument("--vae_checkpoint", type=str, default=None, help="Path to VAE checkpoint (overrides config)")
    parser.add_argument(
        "--diffusion_checkpoint",
        type=str,
        default=None,
        help="Path to diffusion checkpoint (overrides config)",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--metadata_dir", type=str, default="data/metadata", help="Dataset metadata directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "real_world", "synthetic"],
        help="Dataset source",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--augment_test", action="store_true", help="Apply stochastic augmentation on test set")
    parser.add_argument(
        "--num_visualizations",
        type=int,
        default=24,
        help="Number of side-by-side visualizations to save",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return requested


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_model_state(checkpoint: Dict) -> Dict:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _build_components(config: Dict, vae_ckpt: str, diff_ckpt: str, device: str):
    diffusion_cfg = _load_yaml(config["diffusion_config"])

    vae_interface = FrozenVAELatentInterface(
        model_config_path=config["vae_config"],
        checkpoint_path=vae_ckpt,
        device=device,
        use_mu_only=True,
    )

    model = LatentDiffusionUNet(**diffusion_cfg["model"]).to(device)
    checkpoint = torch.load(diff_ckpt, map_location=device)
    model.load_state_dict(_extract_model_state(checkpoint), strict=True)
    model.eval()

    scheduler = LatentDiffusionScheduler(**diffusion_cfg["scheduler"], device=device)
    return vae_interface, model, scheduler


def _ddim_timesteps(num_train_timesteps: int, num_inference_steps: int) -> List[int]:
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    if num_inference_steps > num_train_timesteps:
        raise ValueError("num_inference_steps cannot exceed num_train_timesteps")

    step_ratio = num_train_timesteps / num_inference_steps
    ts = (np.arange(num_inference_steps) * step_ratio).round().astype(np.int64)
    ts = np.clip(ts, 0, num_train_timesteps - 1)
    return ts[::-1].tolist()


@torch.no_grad()
def _sample_refined_latent_ddim(
    model,
    scheduler,
    z_coarse: torch.Tensor,
    num_inference_steps: int,
    eta: float,
) -> torch.Tensor:
    ddim_steps = _ddim_timesteps(scheduler.num_train_timesteps, num_inference_steps)
    z_t = torch.randn_like(z_coarse)

    for idx, step in enumerate(ddim_steps):
        t = torch.full((z_t.shape[0],), step, device=z_t.device, dtype=torch.long)
        eps_pred = model(z_t, t, z_coarse)

        alpha_bar_t = scheduler.alphas_cumprod[step]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        x0_pred = (z_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

        if idx == len(ddim_steps) - 1:
            z_t = x0_pred
            continue

        prev_step = ddim_steps[idx + 1]
        alpha_bar_prev = scheduler.alphas_cumprod[prev_step]

        sigma_t = (
            eta
            * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            * torch.sqrt(1.0 - (alpha_bar_t / alpha_bar_prev))
        )
        sigma_t = torch.clamp(sigma_t, min=0.0)

        pred_dir = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)) * eps_pred
        noise = torch.randn_like(z_t) if eta > 0.0 else torch.zeros_like(z_t)

        z_t = torch.sqrt(alpha_bar_prev) * x0_pred + pred_dir + sigma_t * noise

    return z_t


def _save_sample_figure(rgb, coarse, pred, gt, out_path: Path, sample_id: str, dice: float, iou: float):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(coarse, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Coarse")
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction\nDice={dice:.4f}, IoU={iou:.4f}")
    axes[3].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("GT Refined")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(sample_id)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    print("=" * 50)
    print("Diffusion Inference")
    print("=" * 50)
    print(f"Config: {args.config}")

    config = _load_yaml(args.config)
    inference_cfg = config.get("inference", {})
    vae_ckpt = args.vae_checkpoint or config["vae_checkpoint"]
    diff_ckpt = args.diffusion_checkpoint or config["diffusion_checkpoint"]
    output_dir = Path(args.output_dir or config["output"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)

    device = _resolve_device(config.get("device", "cuda"))

    vae_interface, model, scheduler = _build_components(config, vae_ckpt, diff_ckpt, device)

    image_size = _load_yaml(config["vae_config"])["data"].get("image_size", 512)
    use_aug = args.augment_test
    transform = build_transforms(train=use_aug, augment=use_aug, image_size=image_size)

    dataset = SurgicalMaskRefinementDataset(
        metadata_dir=args.metadata_dir,
        split=args.split,
        source=args.source,
        load_images=True,
        return_paths=False,
        apply_transforms=True,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.get("batch_size", 8),
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    threshold = config["postprocessing"].get("threshold", 0.5)
    num_inference_steps = int(inference_cfg.get("num_inference_steps", 50))
    eta = float(inference_cfg.get("eta", 0.0))

    all_dice: List[float] = []
    all_iou: List[float] = []
    saved_visualizations = 0

    for batch in loader:
        coarse_mask = batch["coarse_mask"].to(device)
        refined_mask = batch["refined_mask"].to(device)

        z_coarse = vae_interface.encode_coarse_mask(coarse_mask)
        z_pred = _sample_refined_latent_ddim(
            model=model,
            scheduler=scheduler,
            z_coarse=z_coarse,
            num_inference_steps=num_inference_steps,
            eta=eta,
        )
        pred_probs = vae_interface.decode_to_probs(z_pred)
        pred_binary = (pred_probs > threshold).float()

        batch_dice = dice_score(pred_probs, refined_mask, threshold=threshold).detach().cpu().tolist()
        batch_iou = iou_score(pred_probs, refined_mask, threshold=threshold).detach().cpu().tolist()
        all_dice.extend(batch_dice)
        all_iou.extend(batch_iou)

        for i in range(pred_binary.shape[0]):
            sample_id = str(batch["id"][i])
            np_pred = (pred_binary[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            out_path = output_dir / "predictions" / f"{sample_id}.png"
            Image.fromarray(np_pred).save(out_path)

            if saved_visualizations < args.num_visualizations:
                rgb_np = batch["rgb"][i].permute(1, 2, 0).cpu().numpy()
                coarse_np = batch["coarse_mask"][i, 0].cpu().numpy()
                gt_np = batch["refined_mask"][i, 0].cpu().numpy()
                pred_np = pred_binary[i, 0].detach().cpu().numpy()
                vis_path = output_dir / "visualizations" / f"{sample_id}_comparison.png"
                _save_sample_figure(
                    rgb=rgb_np,
                    coarse=coarse_np,
                    pred=pred_np,
                    gt=gt_np,
                    out_path=vis_path,
                    sample_id=sample_id,
                    dice=batch_dice[i],
                    iou=batch_iou[i],
                )
                saved_visualizations += 1

    summary = {
        "split": args.split,
        "source": args.source,
        "num_samples": len(all_dice),
        "augment_test": use_aug,
        "sampler": "ddim",
        "num_inference_steps": num_inference_steps,
        "eta": eta,
        "mean_dice": float(np.mean(all_dice)) if all_dice else 0.0,
        "mean_iou": float(np.mean(all_iou)) if all_iou else 0.0,
        "std_dice": float(np.std(all_dice)) if all_dice else 0.0,
        "std_iou": float(np.std(all_iou)) if all_iou else 0.0,
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nInference complete")
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions to: {output_dir / 'predictions'}")
    print(f"Saved visualizations to: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
