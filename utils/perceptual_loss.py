import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tooltipnet import ToolTipNet
from kornia.feature import SOLD2


def tooltipnet_forward_with_features(model: ToolTipNet, x: torch.Tensor):
    x = model.stem(x)
    c2 = model.layer1(x)
    c3 = model.layer2(c2)
    c4 = model.layer3(c3)

    if model.use_attention:
        b, c, h, w = c4.shape
        c4_flat = c4.view(b, c, h * w).permute(0, 2, 1)
        c4_flat = model.positional_encoding(c4_flat)
        c4_flat = model.transformer_encoder(c4_flat)
        c4 = c4_flat.permute(0, 2, 1).view(b, c, h, w)

    c5 = model.layer4(c4)

    features = {"0": c2, "1": c3, "2": c4, "3": c5}
    fpn_out = model.fpn(features)

    p2 = fpn_out["0"]
    p3 = F.interpolate(fpn_out["1"], size=p2.shape[-2:], mode="bilinear", align_corners=False)
    p4 = F.interpolate(fpn_out["2"], size=p2.shape[-2:], mode="bilinear", align_corners=False)
    p5 = F.interpolate(fpn_out["3"], size=p2.shape[-2:], mode="bilinear", align_corners=False)

    fpn_cat = torch.cat([p2, p3, p4, p5], dim=1)
    fpn_feat = model.fpn_fuse(fpn_cat)

    return {
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "c5": c5,
        "fpn_feat": fpn_feat,
    }


class ToolTipFeaturePerceptionLoss(nn.Module):
    def __init__(
        self,
        checkpoint_path: str = "./checkpoints/tooltipnet.pth",
        detector_mask_size: int = 224,
        use_attention: bool = False,
        pretrained_detector: bool = False,
        feature_weights: dict | None = None,
        resize_mode: str = "bilinear",
        loss_type: str = "l2",
    ):
        super().__init__()

        self.detector_mask_size = detector_mask_size
        self.resize_mode = resize_mode
        self.loss_type = loss_type

        self.feature_weights = feature_weights or {
            "c2": 0.,
            "c3": 0.,
            "c4": 0.,
            "c5": 0.,
            "fpn_feat": 1.0,
        }

        self.detector = ToolTipNet(
            mask_size=detector_mask_size,
            pretrained=pretrained_detector,
            use_attention=use_attention,
        )

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.detector.load_state_dict(state_dict)
        self.detector.eval()

        for p in self.detector.parameters():
            p.requires_grad = False # Freeze the detector parameters

    def _prepare_mask(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1:
            x = x[:, :1]

        x = x.float()
        x = torch.clamp(x, 0.0, 1.0)

        if self.resize_mode in ("bilinear", "bicubic"):
            x = F.interpolate(
                x,
                size=(self.detector_mask_size, self.detector_mask_size),
                mode=self.resize_mode,
                align_corners=False,
            )
        else:
            x = F.interpolate(
                x,
                size=(self.detector_mask_size, self.detector_mask_size),
                mode=self.resize_mode,
            )

        return x

    def _feature_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l2":
            return F.mse_loss(a, b)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(a, b)
        return F.l1_loss(a, b)

    def forward(self, pred_mask: torch.Tensor, target_mask: torch.Tensor):
        pred_mask = self._prepare_mask(pred_mask)
        target_mask = self._prepare_mask(target_mask)

        pred_feats = tooltipnet_forward_with_features(self.detector, pred_mask)

        with torch.no_grad():
            target_feats = tooltipnet_forward_with_features(self.detector, target_mask)

        total_loss = pred_mask.new_tensor(0.0)
        loss_dict = {}

        for key, weight in self.feature_weights.items():
            if weight <= 0:
                continue
            feat_loss = self._feature_distance(pred_feats[key], target_feats[key])
            total_loss = total_loss + weight * feat_loss
            loss_dict[f"feature_{key}"] = feat_loss.detach()

        loss_dict["total_loss"] = total_loss.detach()
        return total_loss, loss_dict
    

class SOLD2BackboneFeatureExtractor(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        stage_names=None,
        use_final_backbone_feature: bool = True,
    ):
        super().__init__()

        self.sold2 = SOLD2(pretrained=pretrained)
        self.model = self.sold2.model
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.use_final_backbone_feature = use_final_backbone_feature

        default_stage_names = [
            "backbone_net.net.layer1",
            "backbone_net.net.layer2",
            "backbone_net.net.layer3",
            "backbone_net.net.hg",
        ]

        available = dict(self.model.named_modules())
        # print(available.keys())
        if stage_names is None:
            self.stage_names = [name for name in default_stage_names if name in available]
        else:
            self.stage_names = stage_names

        for name in self.stage_names:
            if name not in available:
                raise ValueError(
                    f"Backbone stage '{name}' not found in SOLD2 model. "
                    f"Available examples: {list(available.keys())[:50]}"
                )

        self._features = {}
        self._hooks = []
        self._register_hooks()

    def _make_hook(self, name):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            if torch.is_tensor(out):
                self._features[name] = out
        return hook

    def _register_hooks(self):
        modules = dict(self.model.named_modules())
        for name in self.stage_names:
            handle = modules[name].register_forward_hook(self._make_hook(name))
            self._hooks.append(handle)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, x: torch.Tensor):
        self._features = {}
        backbone_feature = self.model.backbone_net(x)
        out = dict(self._features)
        if self.use_final_backbone_feature:
            out["backbone_feature"] = backbone_feature
        return out


class SOLD2FeaturePerceptionLoss(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        input_size=(224, 224),
        resize_mode: str = "bilinear",
        loss_type: str = "l2",
        stage_names=None,
        feature_weights=None,
        use_final_backbone_feature: bool = True,
        mean: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.resize_mode = resize_mode
        self.loss_type = loss_type

        self.extractor = SOLD2BackboneFeatureExtractor(
            pretrained=pretrained,
            stage_names=stage_names,
            use_final_backbone_feature=use_final_backbone_feature,
        )
        self.extractor.eval()

        if feature_weights is None:
            feature_weights = {}
            for name in self.extractor.stage_names:
                feature_weights[name] = 1.0
            if use_final_backbone_feature:
                feature_weights["backbone_feature"] = 1.0

        self.feature_weights = feature_weights
        self.compute_mean = mean

    def _prepare_mask(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1:
            x = x[:, :1]

        x = x.float()
        x = torch.clamp(x, 0.0, 1.0)

        if tuple(x.shape[-2:]) != tuple(self.input_size):
            if self.resize_mode in ("bilinear", "bicubic"):
                x = F.interpolate(
                    x,
                    size=self.input_size,
                    mode=self.resize_mode,
                    align_corners=False,
                )
            else:
                x = F.interpolate(
                    x,
                    size=self.input_size,
                    mode=self.resize_mode,
                )

        return x

    def _feature_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l2":
            return F.mse_loss(a, b)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(a, b)
        return F.l1_loss(a, b)

    def forward(self, pred_mask: torch.Tensor, target_mask: torch.Tensor):
        pred_mask = self._prepare_mask(pred_mask)
        target_mask = self._prepare_mask(target_mask)

        pred_feats = self.extractor(pred_mask)

        with torch.no_grad():
            target_feats = self.extractor(target_mask)

        total_loss = pred_mask.new_tensor(0.0)
        loss_dict = {}

        for name, weight in self.feature_weights.items():
            if weight <= 0:
                continue
            if name not in pred_feats or name not in target_feats:
                continue

            # print(pred_feats[name].shape, target_feats[name].shape)

            feat_loss = self._feature_distance(pred_feats[name], target_feats[name])
            total_loss = total_loss + weight * feat_loss
            loss_dict[f"feature_{name}"] = feat_loss.detach()

        loss_dict["total_loss"] = total_loss.detach()
        return total_loss, loss_dict


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    criterion = ToolTipFeaturePerceptionLoss(
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

    # Randomly generate noisy masks for testing
    batch_size = 4

    pred_masks = torch.rand(batch_size, 1, 224, 224).to(device)
    target_masks = torch.rand(batch_size, 1, 224, 224).to(device)

    loss, loss_dict = criterion(pred_masks, target_masks)
    print("ToolTipNet Feature Perception Loss:", loss.item())
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item()}")

    sold2_loss, sold2_loss_dict = sold2_criterion(pred_masks, target_masks)
    print("SOLD2 Feature Perception Loss:", sold2_loss.item())
    for k, v in sold2_loss_dict.items():
        print(f"  {k}: {v.item()}")