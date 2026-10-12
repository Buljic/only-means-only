"""Learned, region-constrained residual editor.

Real shadows have no scalar ``darkness_k``, so the synthetic AutoStrength regressor and
the parametric Lab editors don't transfer. This editor instead *learns* the correction:
a small U-Net predicts a per-pixel RGB residual from ``concat(source, mask)``, and the
result is composited into the source with the **same hard-preserve blend** as the CPU
editor -- so the certified zero-leakage guarantee (Theorem 1) is inherited unchanged:
nothing outside the feathered mask can move, whatever the network predicts.

Device-agnostic (CPU or GPU), so training/inference run unchanged on a GCP GPU.
Trained by ``src/training/train_residual.py``; loaded via ``try_load`` with graceful
fallback (the evaluator simply omits it when no checkpoint exists).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.editor.cpu_editor import composite_local
from src.models.unet import SmallUNet


class ResidualEditor:
    def __init__(self, model: SmallUNet, feather_radius: int = 3,
                 outside_threshold: float = 0.01, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.feather_radius = feather_radius
        self.outside_threshold = outside_threshold

    @classmethod
    def try_load(cls, ckpt_dir: str | Path) -> "ResidualEditor | None":
        try:
            path = Path(ckpt_dir) / "best.pt"
            if not path.exists():
                return None
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = SmallUNet(in_channels=4, out_channels=3, **ckpt.get("model_kwargs", {}))
            model.load_state_dict(ckpt["model_state_dict"])
            return cls(
                model,
                feather_radius=ckpt.get("feather_radius", 3),
                outside_threshold=ckpt.get("outside_threshold", 0.01),
            )
        except Exception:
            return None

    @torch.no_grad()
    def edit(self, source_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Edit ``source_rgb`` (HxWx3 uint8) inside ``mask`` (HxW [0,1]); returns uint8.

        Locality is guaranteed by ``composite_local`` regardless of the model weights.
        """
        src = torch.from_numpy(source_rgb.astype(np.float32) / 255.0)
        src = src.permute(2, 0, 1).unsqueeze(0).to(self.device)          # [1,3,H,W]
        m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

        residual = self.model(torch.cat([src, m], dim=1))               # [1,3,H,W]
        edited = (src + residual).clamp(0.0, 1.0)
        edited_rgb = (edited[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        return composite_local(
            source_rgb, edited_rgb, mask,
            feather_radius=self.feather_radius, outside_threshold=self.outside_threshold,
        )


def build_residual_model(base_channels: int = 32, depth: int = 4) -> SmallUNet:
    """U-Net mapping concat(source_rgb, mask) [B,4,H,W] -> RGB residual [B,3,H,W]."""
    return SmallUNet(in_channels=4, out_channels=3, base_channels=base_channels, depth=depth)
