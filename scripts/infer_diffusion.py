"""Inference script for RGB-conditioned latent diffusion mask refinement.

Runs a trained RGB-conditioned diffusion model on token-conditioned data.
Compatible with train_rgb_conditioned_diffusion.py and
RGBConditionedLatentDiffusionTrainer.

Features:
- Loads frozen VAE + RGB-conditioned diffusion U-Net
- Uses precomputed CLIP RGB tokens via TokenConditionedMaskDataset
- DDIM sampling with RGB conditioning
- Saves binary predictions and side-by-side visualizations
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

from data.token_dataset import TokenConditionedMaskDataset
from models.diffusion import (
    FrozenVAELatentInterface,
    LatentDiffusionScheduler,
    RGBConditionedLatentDiffusionUNet,
)
from utils.metrics import dice_score, iou_score
from utils.perceptual_loss import ToolTipFeaturePerceptionLoss, SOLD2FeaturePerceptionLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Run RGB-conditioned latent diffusion inference")
    parser.add_argument("--config", type=str, required=True, help="Path to inference config YAML")
    parser.add_argument("--vae_checkpoint", type=str, default=None, help="Path to VAE checkpoint")
    parser.add_argument("--diffusion_checkpoint", type=str, default=None, help="Path to diffusion checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--metadata_dir", type=str, default=None, help="Metadata directory override")
    parser.add_argument("--token_dir", type=str, default=None, help="Token directory override")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "real_world", "synthetic"],
        help="Dataset source",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--num_visualizations", type=int, default=24, help="Number of visualizations to save")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Device override")
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


def _strip_common_prefix(key: str) -> str:
    for prefix in ("model.", "module."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _clean_state_dict_for_model(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {_strip_common_prefix(k): v for k, v in state_dict.items()}


def _build_components(config: Dict, vae_ckpt: str, diff_ckpt: str, device: str):
    diffusion_cfg = _load_yaml(config["diffusion_config"])

    vae_interface = FrozenVAELatentInterface(
        model_config_path=config["vae_config"],
        checkpoint_path=vae_ckpt,
        device=device,
        use_mu_only=config.get("vae", {}).get("use_mu_only", True),
    )

    model_cfg = diffusion_cfg["model"]
    rgb_cfg = diffusion_cfg.get("rgb_condition", {})

    model = RGBConditionedLatentDiffusionUNet(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg["base_channels"],
        channel_multipliers=model_cfg["channel_multipliers"],
        num_res_blocks=model_cfg["num_res_blocks"],
        time_embed_dim=model_cfg["time_embed_dim"],
        norm=model_cfg.get("norm", "group"),
        activation=model_cfg.get("activation", "silu"),
        dropout=model_cfg.get("dropout", 0.0),
        rgb_token_dim=rgb_cfg.get("token_dim", 768),
        rgb_projected_dim=rgb_cfg.get("projected_dim", 256),
        rgb_num_heads=rgb_cfg.get("num_heads", 4),
    ).to(device)

    checkpoint = torch.load(diff_ckpt, map_location=device)
    raw_state = _extract_model_state(checkpoint)
    clean_state = _clean_state_dict_for_model(raw_state)

    incompatible = model.load_state_dict(clean_state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "Failed to load RGB-conditioned diffusion checkpoint. "
            f"Missing keys: {incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        print(f"WARNING: Ignoring unexpected checkpoint keys: {incompatible.unexpected_keys}")

    model.eval()

    scheduler_cfg = diffusion_cfg["scheduler"]
    scheduler = LatentDiffusionScheduler(
        num_train_timesteps=scheduler_cfg["num_train_timesteps"],
        beta_schedule=scheduler_cfg["beta_schedule"],
        beta_start=scheduler_cfg["beta_start"],
        beta_end=scheduler_cfg["beta_end"],
        device=device,
    )

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
    rgb_tokens: torch.Tensor,
    num_inference_steps: int,
    eta: float,
) -> torch.Tensor:
    ddim_steps = _ddim_timesteps(scheduler.num_train_timesteps, num_inference_steps)
    z_t = torch.randn_like(z_coarse)

    for idx, step in enumerate(ddim_steps):
        t = torch.full((z_t.shape[0],), step, device=z_t.device, dtype=torch.long)

        # RGB-conditioned prediction
        eps_pred = model(z_t, t, z_coarse, rgb_tokens)

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


def _save_sample_figure(rgb, coarse, pred, gt, out_path: Path, sample_id: str, dice: float, iou: float, coarse_tooltip_loss: float, refined_tooltip_loss: float, coarse_sold2_loss: float, refined_sold2_loss: float):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    if rgb is not None:
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
    else:
        axes[0].axis("off")

    axes[1].imshow(coarse, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Coarse\nToolTip={coarse_tooltip_loss:.4f}, SOLD2={coarse_sold2_loss:.4f}")

    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction\nToolTip={refined_tooltip_loss:.4f}, SOLD2={refined_sold2_loss:.4f}")

    axes[3].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("GT Refined")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(sample_id)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _to_vis_rgb(rgb_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor [3,H,W] to displayable RGB numpy array [H,W,3].
    Assumes tensor is in [0,1] or close to it.
    """
    rgb_np = rgb_tensor.detach().cpu().permute(1, 2, 0).numpy()
    rgb_np = np.clip(rgb_np, 0.0, 1.0)
    return rgb_np


def _compute_batch_score(criterion, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute perceptual loss for a batch of predictions and ground truths.

    Args:
        criterion: Perceptual loss module (e.g. ToolTipFeaturePerceptionLoss)
        pred: Tensor of shape [B,1,H,W] with binary predictions
        gt: Tensor of shape [B,1,H,W] with binary ground truths
    Returns:
        Tensor of shape [B] with loss values for each sample in the batch
    """
    batch_size = pred.shape[0]
    losses = []
    for i in range(batch_size):
        loss = criterion(pred[i:i+1], gt[i:i+1])[0]
        losses.append(loss)
    return torch.stack(losses).squeeze()


def main():
    args = parse_args()

    config = _load_yaml(args.config)
    inference_cfg = config.get("inference", {})

    device = _resolve_device(args.device or config.get("device", "cuda"))

    tooltip_criterion = ToolTipFeaturePerceptionLoss(
        checkpoint_path="./checkpoints/tooltipnet.pth",
        detector_mask_size=224,
        use_attention=False,
        feature_weights={
            "c2": 0.,
            "c3": 0.,
            "c4": 0.,
            "c5": 0.,
            "fpn_feat": 1.0,
        },
        loss_type="l2",
    ).to(device)

    # SOLD2
    sold2_criterion = SOLD2FeaturePerceptionLoss(
        pretrained=True,
        input_size=(224, 224),
        loss_type="l2",
    ).to(device)

    print("=" * 60)
    print("RGB-CONDITIONED LATENT DIFFUSION INFERENCE")
    print("=" * 60)
    print(f"Config: {args.config}")

    vae_ckpt = args.vae_checkpoint or config["vae_checkpoint"]
    diff_ckpt = args.diffusion_checkpoint or config["diffusion_checkpoint"]
    output_dir = Path(args.output_dir or config["output"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)

    vae_interface, model, scheduler = _build_components(config, vae_ckpt, diff_ckpt, device)

    # Pull dataset settings from training config if present
    data_cfg = config.get("data", {})
    metadata_dir = args.metadata_dir or data_cfg.get("metadata_dir", "data/metadata")
    token_dir = args.token_dir or data_cfg.get("token_dir", "outputs/clip_tokens")
    image_size = data_cfg.get("image_size", 512)
    strict_tokens = data_cfg.get("strict_tokens", True)

    dataset = TokenConditionedMaskDataset(
        metadata_dir=metadata_dir,
        token_dir=token_dir,
        split=args.split,
        source=args.source,
        image_size=image_size,
        load_spatial_map=False,
        return_paths=True,
        strict_tokens=strict_tokens,
        apply_augmentation=True,  # no augmentation during inference
        transform=None,  # keep deterministic, matching validation/test style
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.get("batch_size", 8),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    threshold = config.get("postprocessing", {}).get("threshold", 0.5)
    num_inference_steps = int(inference_cfg.get("num_inference_steps", 50))
    eta = float(inference_cfg.get("eta", 0.0))

    all_dice: List[float] = []
    all_iou: List[float] = []
    all_tooltip_losses: List[float] = []
    all_sold2_losses: List[float] = []
    saved_visualizations = 0

    for batch in loader:
        coarse_mask = batch["coarse_mask"].to(device)      # [B,1,512,512]
        refined_mask = batch["refined_mask"].to(device)    # [B,1,512,512]
        rgb_tokens = batch["rgb_tokens"].to(device)        # [B,196,768]

        with torch.no_grad():
            z_coarse = vae_interface.encode_coarse_mask(coarse_mask)

            z_pred = _sample_refined_latent_ddim(
                model=model,
                scheduler=scheduler,
                z_coarse=z_coarse,
                rgb_tokens=rgb_tokens,
                num_inference_steps=num_inference_steps,
                eta=eta,
            )

            pred_probs = vae_interface.decode_to_probs(z_pred)
            pred_binary = (pred_probs > threshold).float()

        batch_dice = dice_score(pred_probs, refined_mask, threshold=threshold).detach().cpu().tolist()
        batch_iou = iou_score(pred_probs, refined_mask, threshold=threshold).detach().cpu().tolist()
        batch_coarse_tooltip_loss = _compute_batch_score(tooltip_criterion, coarse_mask, refined_mask).detach().cpu().tolist()
        batch_coarse_sold2_loss = _compute_batch_score(sold2_criterion, coarse_mask, refined_mask).detach().cpu().tolist()
        batch_tooltip_loss = _compute_batch_score(tooltip_criterion, pred_binary, refined_mask).detach().cpu().tolist()
        batch_sold2_loss = _compute_batch_score(sold2_criterion, pred_binary, refined_mask).detach().cpu().tolist()
        all_dice.extend(batch_dice)
        all_iou.extend(batch_iou)
        all_tooltip_losses.extend(batch_tooltip_loss)
        all_sold2_losses.extend(batch_sold2_loss)

        for i in range(pred_binary.shape[0]):
            sample_id = str(batch["id"][i])
            refined_mask_path = batch["refined_mask_path"][i]
            rgb_path = refined_mask_path.replace("refined_mask", "RGB")
            # print(rgb_path)
            rgb_np = np.array(Image.open(rgb_path).convert("RGB").resize((image_size, image_size)))

            np_pred = (pred_binary[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            out_path = output_dir / "predictions" / f"{sample_id}.png"

            rgb_overlay = rgb_np.copy().astype(np.uint8)
            mask = np_pred > 0
            rgb_overlay[mask] = (rgb_overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

            Image.fromarray(rgb_overlay).save(out_path)

            if saved_visualizations < args.num_visualizations:
                coarse_np = batch["coarse_mask"][i, 0].detach().cpu().numpy()
                gt_np = batch["refined_mask"][i, 0].detach().cpu().numpy()
                pred_np = pred_binary[i, 0].detach().cpu().numpy()

                coarse_overlay = rgb_np.copy()
                mask = coarse_np > 0
                coarse_overlay[mask] = (coarse_overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)  # green overlay for coarse mask

                refined_overlay = rgb_np.copy()
                mask = gt_np > 0
                refined_overlay[mask] = (refined_overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)  # green overlay for refined GT

                vis_path = output_dir / "visualizations" / f"{sample_id}_comparison.png"
                _save_sample_figure(
                    rgb=rgb_np,
                    coarse=coarse_overlay,
                    pred=rgb_overlay,
                    gt=refined_overlay,
                    out_path=vis_path,
                    sample_id=sample_id,
                    dice=batch_dice[i],
                    iou=batch_iou[i],
                    coarse_tooltip_loss=batch_coarse_tooltip_loss[i],
                    coarse_sold2_loss=batch_coarse_sold2_loss[i],
                    refined_tooltip_loss=batch_tooltip_loss[i],
                    refined_sold2_loss=batch_sold2_loss[i],
                )
                saved_visualizations += 1

    summary = {
        "split": args.split,
        "source": args.source,
        "num_samples": len(all_dice),
        "sampler": "ddim",
        "num_inference_steps": num_inference_steps,
        "eta": eta,
        "mean_dice": float(np.mean(all_dice)) if all_dice else 0.0,
        "mean_iou": float(np.mean(all_iou)) if all_iou else 0.0,
        "std_dice": float(np.std(all_dice)) if all_dice else 0.0,
        "std_iou": float(np.std(all_iou)) if all_iou else 0.0,
        "mean_tooltip_loss": float(np.mean(all_tooltip_losses)) if all_tooltip_losses else 0.0,
        "mean_sold2_loss": float(np.mean(all_sold2_losses)) if all_sold2_losses else 0.0,
    }

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nInference complete")
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions to: {output_dir / 'predictions'}")
    print(f"Saved visualizations to: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()