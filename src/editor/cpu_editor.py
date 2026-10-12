"""Deterministic CPU shadow removal editor with zero-leakage guarantee."""

import numpy as np
from scipy.ndimage import gaussian_filter

from src.utils.image_utils import rgb_to_lab, lab_to_rgb


def feather_mask(mask: np.ndarray, feather_radius: int) -> np.ndarray:
    """Apply Gaussian blur to soften mask edges. Input/output: float32 [0,1]."""
    if feather_radius <= 0:
        return mask
    feathered = gaussian_filter(mask.astype(np.float32), sigma=feather_radius)
    return np.clip(feathered, 0, 1).astype(np.float32)


def _hard_preserve_blend(source_rgb, edited_rgb, mask_feathered, outside_threshold):
    """Blend ``edited_rgb`` into ``source_rgb`` weighted by the feathered mask, with a
    hard zero-leakage preserve where ``mask_feathered < outside_threshold``.

    This is THE single place the locality guarantee lives (Theorem 1): pixels with
    ``mask_feathered < outside_threshold`` get blend weight 0, so result == source there.
    """
    m = mask_feathered[:, :, np.newaxis]
    blend = np.where(m < outside_threshold, 0.0, m.clip(outside_threshold, 1.0))
    result = blend * edited_rgb.astype(np.float32) + (1.0 - blend) * source_rgb.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def composite_local(source_rgb, edited_rgb, mask, feather_radius=3, outside_threshold=0.01):
    """Feather ``mask`` and hard-preserve-blend an arbitrary ``edited_rgb`` into the
    source. Lets any editor (e.g. the learned residual editor) inherit the exact same
    certified zero-leakage guarantee as the CPU editor.
    """
    return _hard_preserve_blend(
        source_rgb, edited_rgb, feather_mask(mask, feather_radius), outside_threshold
    )


def apply_shadow_removal(
    source_rgb: np.ndarray,
    mask: np.ndarray,
    strength: float,
    feather_radius: int = 3,
    max_boost: float = 50.0,
    outside_threshold: float = 0.01,
    mode: str = "additive",
) -> np.ndarray:
    """Remove shadow by correcting luminance inside the mask region.

    Two correction models share the same feather + hard-preserve machinery:

    - ``"additive"`` (default, original behaviour):
          L_edit = L + strength * mask * max_boost
      A heuristic luminance boost. Cheap, but it is *not* the inverse of the
      synthetic forward model, so it over/under-corrects depending on L.

    - ``"divisive"``: the analytic inverse of the synthetic shadow model
      ``L_src = L * (1 - k * mask)`` used by the dataset generator. Recovering the
      clean luminance is a division, using the predicted strength as k:
          L_edit = L / (1 - clip(strength * mask, 0, 0.95))
      Parameter-free (no max_boost) and exact when strength == k.

    Common steps: feather mask -> Lab -> correct L -> back to RGB -> hard-preserve
    pixels with mask < outside_threshold (zero-leakage guarantee).

    Args:
        source_rgb: HxWx3 uint8, image with shadow.
        mask: HxW float32 [0,1], predicted shadow region.
        strength: float [0,1], predicted shadow intensity.
        feather_radius: Gaussian blur radius for edge softening.
        max_boost: maximum Lab L-channel boost at full strength (additive mode only).
        outside_threshold: pixels below this mask value are hard-preserved.
        mode: "additive" or "divisive".

    Returns:
        edited_rgb: HxWx3 uint8, shadow-removed image.
    """
    mask_feathered = feather_mask(mask, feather_radius)

    lab = rgb_to_lab(source_rgb)

    L = lab[:, :, 0]
    if mode == "additive":
        L_edit = L + strength * mask_feathered * max_boost
    elif mode == "divisive":
        # Guard the denominator: strength*mask in [0, 0.95] -> at most ~20x gain.
        atten = np.clip(strength * mask_feathered, 0.0, 0.95)
        L_edit = L / (1.0 - atten)
    else:
        raise ValueError(f"Unknown editor mode: {mode!r} (expected 'additive' or 'divisive')")
    L_edit = np.clip(L_edit, 0, 100)
    lab[:, :, 0] = L_edit

    edited_rgb = lab_to_rgb(lab)

    # Hard preserve outside region: blend back original RGB pixels (avoids Lab roundtrip
    # error outside the mask and gives the zero-leakage guarantee).
    return _hard_preserve_blend(source_rgb, edited_rgb, mask_feathered, outside_threshold)


class CPUEditor:
    """Stateless shadow removal editor wrapping apply_shadow_removal.

    ``mode`` selects the luminance-correction model ("additive" or "divisive").
    If not given explicitly it falls back to ``cfg_editor.mode`` when present,
    otherwise "additive".
    """

    def __init__(self, cfg_editor, mode: str | None = None):
        self.feather_radius = cfg_editor.feather_radius
        self.max_boost = cfg_editor.max_boost
        self.outside_threshold = cfg_editor.outside_threshold
        self.mode = mode or getattr(cfg_editor, "mode", "additive")

    def edit(self, source_rgb: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
        """Edit source image: remove shadow inside mask at given strength."""
        return apply_shadow_removal(
            source_rgb=source_rgb,
            mask=mask,
            strength=strength,
            feather_radius=self.feather_radius,
            max_boost=self.max_boost,
            outside_threshold=self.outside_threshold,
            mode=self.mode,
        )
