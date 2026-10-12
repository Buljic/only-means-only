"""Certified-locality toolkit.

This module operationalizes the claim that the editor changes *only* the intended
region. It provides:

1. ``leakage_report`` - rigorous out-of-region change under several norms
   (L1, L-infinity, and support fraction). L-infinity and support are the
   discriminative ones: a certified-local editor drives them to 0; a generative /
   global editor does not.
2. ``LocalityCertificate`` - encodes the zero-leakage theorem for a given editor
   configuration and returns the set of pixels guaranteed byte-identical.
3. ``verify_zero_leakage`` - a runtime verifier that *checks* the guarantee held on
   a concrete (source, edited) pair (max violation on the certified set must be 0).
4. ``compare_editors`` - a harness that measures leakage for any set of editors
   (callables ``(source, mask, strength) -> edited``), so a real diffusion / instruct
   editor can be dropped in next to ours.
5. Stand-in "leaky" editors (global relight / global noise) that simulate how an
   uncertified generative edit perturbs the whole frame, for demonstration until a
   real model is wired in.

The zero-leakage guarantee is *structural* (it does not depend on the trained
networks), which is exactly what makes it certifiable. The statements and proofs are
given in the class and function docstrings below.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.editor.cpu_editor import CPUEditor, composite_local, feather_mask
from src.utils.image_utils import lab_to_rgb, rgb_to_lab

# An editor is any callable (source_rgb, mask, strength) -> edited_rgb.
EditorFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


# --------------------------------------------------------------------------- #
# Leakage metric
# --------------------------------------------------------------------------- #
def leakage_report(
    original: np.ndarray,
    edited: np.ndarray,
    region_mask: np.ndarray,
    threshold: float = 0.5,
    eps: float = 0.0,
) -> dict:
    """Quantify change *outside* the intended edit region.

    Args:
        original / edited: HxWx3 uint8 RGB.
        region_mask: HxW in [0,1]; "outside" = pixels with ``mask < threshold``.
        threshold: region cutoff (0.5 for a hard {0,1} mask).
        eps: a pixel counts as "changed" if its max-channel abs diff > eps.

    Returns dict with:
        n_outside, l1 (mean abs, channel-averaged), linf (worst single channel
        deviation anywhere outside), support_frac (fraction of outside pixels changed),
        changed_pixels.
    """
    outside = region_mask < threshold
    n_outside = int(outside.sum())
    if n_outside == 0:
        return {"n_outside": 0, "l1": 0.0, "linf": 0.0, "support_frac": 0.0, "changed_pixels": 0}

    diff = np.abs(edited.astype(np.float32) - original.astype(np.float32))
    if diff.ndim == 3:
        diff_inf = diff.max(axis=2)   # worst channel per pixel
        diff_l1 = diff.mean(axis=2)   # channel-averaged per pixel
    else:
        diff_inf = diff
        diff_l1 = diff

    out_inf = diff_inf[outside]
    changed = int((out_inf > eps).sum())
    return {
        "n_outside": n_outside,
        "l1": float(diff_l1[outside].mean()),
        "linf": float(out_inf.max()),
        "support_frac": float(changed / n_outside),
        "changed_pixels": changed,
    }


def background_metrics(
    original: np.ndarray,
    edited: np.ndarray,
    region_mask: np.ndarray,
    feather_radius: int = 3,
    outside_threshold: float = 0.01,
    eps_perceptual: float = 2.0,
) -> dict:
    """Field-standard *background-preservation* footprint metrics (PIE-Bench / GIE-Bench style).

    Two corrections over the raw ``leakage_report`` support count, both of which a
    defensible footprint metric needs:

    1. **Measure outside the *feathered* edit support**, not the hard mask. A smooth edit
       boundary (the feather halo) is part of the intended region; counting it as leakage
       penalises every well-behaved local editor (including ours). "Background" here is the
       set the zero-leakage theorem certifies unchanged: ``feather(mask) < outside_threshold``.
    2. **Report magnitude, not a binary support@eps=0 count.** A binary "any change" count is
       dominated by sub-perceptual VAE/resize round-trip noise: a diffusion editor that
       re-encodes the whole frame trips ~100% of pixels by +/-1 even where it changed nothing
       meaningful. PIE-Bench instead reports background PSNR/MSE/LPIPS magnitude.

    Returns:
        bg_psnr   : PSNR (dB) over the background only (higher = better; capped 99).
        bg_mse    : mean squared error over the background (lower = better).
        bg_l1     : mean abs diff over the background.
        support_eps: fraction of background pixels changed by more than ``eps_perceptual``
                     (the honest, perception-thresholded footprint; ~0 for a local editor).
        support_raw: fraction changed by >0 (the eps=0 number, kept for transparency so the
                     VAE/resize inflation is visible rather than hidden).
        n_background: size of the background set.
    """
    m_feathered = feather_mask(region_mask.astype(np.float32), feather_radius)
    background = m_feathered < outside_threshold
    n_bg = int(background.sum())
    if n_bg == 0:
        return {"bg_psnr": 99.0, "bg_mse": 0.0, "bg_l1": 0.0,
                "support_eps": 0.0, "support_raw": 0.0, "n_background": 0}

    diff = np.abs(edited.astype(np.float32) - original.astype(np.float32))
    diff_inf = diff.max(axis=2) if diff.ndim == 3 else diff
    diff_l1 = diff.mean(axis=2) if diff.ndim == 3 else diff
    sq = (edited.astype(np.float32) - original.astype(np.float32)) ** 2
    bg_mse = float(sq[background].mean()) if sq.ndim == 2 else float(sq[background].mean())

    out_inf = diff_inf[background]
    return {
        "bg_psnr": 99.0 if bg_mse < 1e-8 else float(10.0 * np.log10((255.0 ** 2) / bg_mse)),
        "bg_mse": bg_mse,
        "bg_l1": float(diff_l1[background].mean()),
        "support_eps": float(int((out_inf > eps_perceptual).sum()) / n_bg) * 100.0,
        "support_raw": float(int((out_inf > 0).sum()) / n_bg) * 100.0,
        "n_background": n_bg,
    }


# --------------------------------------------------------------------------- #
# Certificate + verifier
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LocalityCertificate:
    """Encodes the zero-leakage theorem for the CPU editor.

    Theorem (zero-leakage). With hard-preserve threshold ``tau`` and feather radius
    ``r``, every pixel ``p`` with ``feather(mask, r)(p) < tau`` satisfies
    ``edited(p) == source(p)`` exactly, for *any* strength and *any* network outputs,
    because the blend weight there is 0 (``result = 0*edited_raw + 1*source``).
    """

    threshold: float
    feather_radius: int
    mode: str = "additive"

    def guaranteed_unchanged(self, used_mask: np.ndarray) -> np.ndarray:
        """Boolean HxW map of pixels the certificate guarantees are byte-identical.

        ``used_mask`` must be the mask actually passed to the editor (predicted in the
        pipeline, GT in the oracle setting); the guarantee is relative to its feathered
        support.
        """
        m_f = feather_mask(used_mask.astype(np.float32), self.feather_radius)
        return m_f < self.threshold

    def certified_fraction(self, used_mask: np.ndarray) -> float:
        """Fraction of the image the certificate guarantees is unchanged."""
        return float(self.guaranteed_unchanged(used_mask).mean())


def verify_zero_leakage(
    source: np.ndarray,
    edited: np.ndarray,
    used_mask: np.ndarray,
    threshold: float,
    feather_radius: int,
) -> tuple[bool, float]:
    """Runtime check that the certified-unchanged pixels are byte-identical.

    Returns (ok, max_violation). For a certified-local editor, max_violation == 0.
    """
    cert = LocalityCertificate(threshold, feather_radius)
    guaranteed = cert.guaranteed_unchanged(used_mask)
    if guaranteed.sum() == 0:
        return True, 0.0
    d = np.abs(edited.astype(np.int16) - source.astype(np.int16))
    if d.ndim == 3:
        d = d.max(axis=2)
    violation = float(d[guaranteed].max())
    return violation == 0.0, violation


# --------------------------------------------------------------------------- #
# Comparison harness + stand-in "leaky" editors
# --------------------------------------------------------------------------- #
def compare_editors(
    source: np.ndarray,
    mask: np.ndarray,
    strength: float,
    editors: dict[str, EditorFn],
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Run each editor on (source, mask, strength) and report out-of-region leakage.

    ``mask`` is the *intended* region all editors are asked to edit; leakage is the
    change they induce outside it. Plug a real diffusion/instruct editor in as a
    callable to compare it against the certified-local editor.
    """
    return {
        name: leakage_report(source, fn(source, mask, strength), mask, threshold)
        for name, fn in editors.items()
    }


def make_cpu_editor_fn(cfg_editor, mode: str) -> EditorFn:
    """Wrap the certified-local CPU editor as an EditorFn."""
    editor = CPUEditor(cfg_editor, mode=mode)
    return lambda s, m, k: editor.edit(s, m, k)


def make_global_relight_editor(boost: float = 25.0) -> EditorFn:
    """Stand-in leaky editor: brightens the WHOLE image, ignoring the mask.

    Mimics a global generative/illumination edit that perturbs every pixel.
    """
    def _edit(source: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
        lab = rgb_to_lab(source)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + strength * boost, 0, 100)
        return lab_to_rgb(lab)

    return _edit


def make_global_context_editor(pull: float = 0.6) -> EditorFn:
    """Stand-in *sensitivity-violating* editor: the in-region edit depends on OUT-of-region
    content (it pulls the masked region toward the mean color of the *outside*).

    Unlike the per-pixel leaky editors (which break footprint but are sensitivity-local),
    this is the right stand-in for a global-attention editor: changing the background
    provably changes the foreground edit. Used to validate the sensitivity audit.
    """
    def _edit(source: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
        m = (mask > 0.5)
        out = ~m
        if out.sum() == 0:
            return source.copy()
        bg_mean = source[out].mean(axis=0)                       # depends on outside pixels
        edited = source.astype(np.float32).copy()
        edited[m] = (1 - pull) * edited[m] + pull * bg_mean
        return np.clip(edited, 0, 255).astype(np.uint8)

    return _edit


def make_certified_local_wrapper(edit_fn: EditorFn, margin: int = 8,
                                 fill: float | None = None) -> EditorFn:
    """Confer a DETERMINISTIC sensitivity certificate on ANY editor.

    Pixels outside the trusted window ``W = dilate(region, margin)`` are replaced by a
    constant (depending only on ``W``) before the editor sees them, so they have
    *provably zero* influence on the in-region edit -- a deterministic guarantee for an
    arbitrary (even diffusion) editor, trading the margin (context) for locality. Footprint
    is preserved by ``composite_local``. This is the answer to "you certified a toy editor":
    wrap a strong editor and certify its locality, paying only the margin.
    """
    from src.utils.masks import dilate_window

    def _edit(source: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
        W = dilate_window(mask, margin)
        fill_val = np.full(3, fill, np.float32) if fill is not None else source[W].mean(axis=0)
        masked = source.astype(np.float32).copy()
        masked[~W] = fill_val                                   # depends only on W
        edited = edit_fn(np.clip(masked, 0, 255).astype(np.uint8), mask, strength)
        return composite_local(source, edited, mask, feather_radius=3, outside_threshold=0.01)

    return _edit


def audit_editor_sensitivity(
    edit_fn: EditorFn, source: np.ndarray, mask: np.ndarray, strength: float = 0.5,
    eps: float = 8.0, n_random: int = 16, spsa_steps: int = 20, seed: int = 0,
    seed_noise_samples: int = 0, perturb_mask: np.ndarray | None = None,
) -> dict:
    """Black-box out-of-region trigger audit for ANY editor.

    Perturbs only allowed pixels by an L-infinity-bounded ``eps`` (in 0..255 pixel units,
    i.e. "imperceptible") and measures the induced change in the *in-region* edit. By
    default the allowed set is the hard-mask background. Passing ``perturb_mask`` makes the
    threat model explicit; for GLW certificates it should be ``not dilate(region, halo)``
    because the halo is trusted local context, not certified background.

    Reports the worst random and worst (SPSA-)adversarial influence (mean per-pixel
    |edit(I+delta) - edit(I)| over the masked region). Editor-agnostic and gradient-free,
    so diffusion / instruct editors plug in directly via the EditorFn interface.

    ``seed_noise_samples > 0`` adds a *sampling-noise floor*: the in-region variation from
    re-running the editor at ``delta=0``. For stochastic (diffusion) editors a trigger is
    only real when ``adv_influence >> seed_noise``; for deterministic editors it is 0.
    """
    rng = np.random.default_rng(seed)
    inside = (mask > 0.5).astype(np.float32)
    if perturb_mask is None:
        # Footgun guard: the default hard-mask complement INCLUDES the trusted halo, so it is the
        # LOOSE diagnostic threat model -- NOT the one the GLW certificate is stated against
        # (Omega = not dilate(region, halo)). A published table must pass an explicit
        # ``perturb_mask``. Warn loudly so a hard-mask number never silently becomes a
        # headline number compared against the certificate.
        import warnings
        warnings.warn(
            "audit_editor_sensitivity: perturb_mask=None -> using the hard-mask complement "
            "(includes the halo). This is the DIAGNOSTIC threat model; do NOT compare it to the "
            "GLW certificate. Pass perturb_mask=~dilate(region, halo) for the certified threat model.",
            stacklevel=2,
        )
        allowed = 1.0 - inside
    else:
        allowed = (perturb_mask > 0.5).astype(np.float32) * (1.0 - inside)
    outside3 = allowed[:, :, None]
    base = edit_fn(source, mask, strength).astype(np.float32)
    denom = inside.sum() + 1e-8

    def influence(delta):
        pert = np.clip(source.astype(np.float32) + delta * outside3, 0, 255).astype(np.uint8)
        diff = np.abs(edit_fn(pert, mask, strength).astype(np.float32) - base).mean(axis=2)
        return float((diff * inside).sum() / denom)

    rand_best, best_delta = 0.0, np.zeros_like(source, dtype=np.float32)
    for _ in range(n_random):
        d = rng.uniform(-eps, eps, source.shape).astype(np.float32)
        v = influence(d)
        if v > rand_best:
            rand_best, best_delta = v, d

    # Structured low-frequency candidates: uniform (per-channel) background shifts. These
    # are realistic "imperceptible" triggers that high-frequency random/SPSA search misses
    # (and catch mean/brightness dependence). They count toward the adversarial estimate.
    adv_best, best_adv_delta = rand_best, best_delta.copy()
    structured = [np.full(source.shape, eps, np.float32), np.full(source.shape, -eps, np.float32)]
    for ch in range(3):
        for s in (eps, -eps):
            d = np.zeros(source.shape, np.float32)
            d[:, :, ch] = s
            structured.append(d)
    for d in structured:
        v = influence(d)
        if v > adv_best:
            adv_best, best_adv_delta = v, d

    delta = best_adv_delta.copy()
    c, lr = eps * 0.5, eps * 0.5
    for _ in range(spsa_steps):
        sgn = rng.choice([-1.0, 1.0], size=source.shape).astype(np.float32)
        g = (influence(delta + c * sgn) - influence(delta - c * sgn)) / (2 * c) * sgn
        delta = np.clip(delta + lr * np.sign(g), -eps, eps)
        adv_best = max(adv_best, influence(delta))

    out = {"random_influence": rand_best, "adv_influence": adv_best, "eps": eps}
    if seed_noise_samples > 0:
        sn = [influence(np.zeros_like(source, dtype=np.float32)) for _ in range(seed_noise_samples)]
        out["seed_noise"] = float(np.mean(sn))
    return out


def make_global_noise_editor(sigma: float = 2.0, seed: int = 0) -> EditorFn:
    """Stand-in leaky editor: adds small global noise everywhere.

    Mimics how a diffusion model resamples the entire frame (tiny but ubiquitous
    changes), which a certified-local editor avoids by construction.
    """
    rng = np.random.default_rng(seed)

    def _edit(source: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
        noise = rng.normal(0.0, sigma, size=source.shape)
        return np.clip(source.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return _edit


# --------------------------------------------------------------------------- #
# Report (CLI): measure leakage of certified-local vs leaky editors on the test set
# --------------------------------------------------------------------------- #
def run_comparison(cfg, max_samples: int | None = None) -> dict:
    """Aggregate leakage over the test split for certified vs leaky editors.

    Model-free: uses GT mask + GT darkness_k as the intended edit, since leakage is a
    structural property of the editor, not of the estimator.
    """
    from src.data.dataset import ShadowDataset
    from src.utils.image_utils import tensor_to_image

    editors = {
        "ours_additive": make_cpu_editor_fn(cfg.editor, "additive"),
        "ours_divisive": make_cpu_editor_fn(cfg.editor, "divisive"),
        "leaky_relight": make_global_relight_editor(),
        "leaky_noise": make_global_noise_editor(),
    }

    ds = ShadowDataset(Path(cfg.data.dataset_dir) / cfg.eval.test_split, augment=False)
    n = len(ds) if max_samples is None else min(max_samples, len(ds))

    acc = {name: {"l1": [], "linf": [], "support_frac": []} for name in editors}
    certified_pass = {name: 0 for name in editors}

    for i in range(n):
        sample = ds[i]
        source = tensor_to_image(sample["source"])
        mask = sample["mask"][0].numpy()  # HxW {0,1}
        strength = float(sample["darkness_k"].item())

        for name, fn in editors.items():
            edited = fn(source, mask, strength)
            rep = leakage_report(source, edited, mask, threshold=0.5)
            for k in ("l1", "linf", "support_frac"):
                acc[name][k].append(rep[k])
            ok, _ = verify_zero_leakage(
                source, edited, mask, cfg.editor.outside_threshold, cfg.editor.feather_radius
            )
            certified_pass[name] += int(ok)

    summary = {"n_samples": n, "editors": {}}
    for name in editors:
        summary["editors"][name] = {
            "mean_leakage_l1": float(np.mean(acc[name]["l1"])),
            "mean_leakage_linf": float(np.mean(acc[name]["linf"])),
            "mean_support_frac": float(np.mean(acc[name]["support_frac"])),
            "certified_zero_leakage_rate": certified_pass[name] / max(n, 1),
        }
    return summary


def plot_comparison(summary: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(summary["editors"].keys())
    support = [summary["editors"][n]["mean_support_frac"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["tab:green" if n.startswith("ours") else "tab:red" for n in names]
    ax.bar(names, support, color=colors)
    ax.set_ylabel("Out-of-region pixels changed (%)")
    ax.set_title("Edit leakage: certified-local (green) vs leaky (red)")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "leakage_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Certified-locality leakage comparison")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="limit number of test samples (default: all)")
    args = parser.parse_args()

    from src.utils.config import load_config

    cfg = load_config(args.config)
    summary = run_comparison(cfg, max_samples=args.max_samples)

    output_dir = Path(cfg.outputs.eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "leakage_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_comparison(summary, output_dir)

    print(f"\nLeakage comparison ({summary['n_samples']} samples)")
    for name, m in summary["editors"].items():
        print(
            f"  [{name:14s}] L1={m['mean_leakage_l1']:.3f}  "
            f"Linf={m['mean_leakage_linf']:.2f}  "
            f"support={m['mean_support_frac'] * 100:6.2f}%  "
            f"certified-0-rate={m['certified_zero_leakage_rate'] * 100:5.1f}%"
        )
    print(f"  Results saved to {output_dir}")


if __name__ == "__main__":
    main()
