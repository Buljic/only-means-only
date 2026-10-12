"""Certified Edit Sensitivity: a machine-checkable certificate for an editor.

Two provable bounds form the certificate:

1. **Footprint** (Theorem 1, via the hard-preserve composite): pixels outside the
   feathered region are byte-identical to the source. See ``src/eval/locality.py``.
2. **Sensitivity**: a sound Lipschitz bound ``L`` such that perturbing the
   out-of-region input by ``delta`` (in L2 over pixels) changes the in-region edit by at
   most ``L * delta``. Diffusion / global-attention editors have no finite a-priori
   bound here -- this is the guarantee they cannot provide.

This module computes the sensitivity certificate for a Lipschitz editor net and provides
an empirical verifier that the certificate is respected (a sanity check, since the bound
is sound by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.lipschitz import is_circular, model_certified_lipschitz


@dataclass(frozen=True)
class SensitivityCertificate:
    """Certified bound on how much out-of-region input can move the in-region edit."""

    lipschitz_bound: float          # L: ||edit(x+d) - edit(x)|| <= L * ||d||
    n_conv_layers: int
    tight: bool = False             # per-layer EXACT (circular-conv FFT operator norm, Sedghi'19).
                                    # NOTE: this flags per-LAYER exactness only; the global bound is
                                    # still a sub-multiplicative PRODUCT of layer norms and can be
                                    # loose end-to-end. Do not report it as a globally tight L.

    def max_edit_change(self, perturbation_l2: float) -> float:
        """Certified worst-case L2 change of the residual for an input perturbation."""
        return self.lipschitz_bound * perturbation_l2


def certify_sensitivity(model: torch.nn.Module, input_size: int | None = None) -> SensitivityCertificate:
    """Sound sensitivity certificate. With ``input_size`` and a circular-padded model it is
    *tight* (exact per-layer FFT operator norm); otherwise it is the loose-but-sound sum bound."""
    n_conv = sum(1 for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    tight = input_size is not None and is_circular(model)
    return SensitivityCertificate(
        lipschitz_bound=model_certified_lipschitz(model, input_size),
        n_conv_layers=n_conv,
        tight=tight,
    )


@torch.no_grad()
def empirical_sensitivity(model: torch.nn.Module, x: torch.Tensor, n_trials: int = 16,
                          eps: float = 0.1, seed: int = 0) -> float:
    """Largest observed ||f(x+d) - f(x)|| / ||d|| over random perturbations.

    Must not exceed the certified Lipschitz bound (the certificate is sound).
    """
    g = torch.Generator(device=x.device).manual_seed(seed)
    model.eval()
    base = model(x)
    worst = 0.0
    for _ in range(n_trials):
        d = torch.randn(x.shape, generator=g, device=x.device)
        d = eps * d / (d.flatten().norm() + 1e-12)
        ratio = (model(x + d) - base).flatten().norm().item() / (d.flatten().norm().item() + 1e-12)
        worst = max(worst, ratio)
    return worst


def verify_sensitivity_certificate(model: torch.nn.Module, x: torch.Tensor,
                                   n_trials: int = 16, eps: float = 0.1) -> tuple[bool, float, float]:
    """Empirically confirm observed sensitivity <= certified bound.

    Returns (ok, empirical, certified). ``ok`` should always hold (soundness); it would
    only fail if the bound were computed incorrectly.
    """
    cert = certify_sensitivity(model)
    emp = empirical_sensitivity(model, x, n_trials=n_trials, eps=eps)
    return emp <= cert.lipschitz_bound + 1e-4, emp, cert.lipschitz_bound


def adversarial_out_of_region_influence(
    model: torch.nn.Module, source_rgb: np.ndarray, mask: np.ndarray,
    eps: float = 0.03, n_steps: int = 40, step: float = 0.005, norm: str = "linf", seed: int = 0,
) -> dict:
    """PGD-witnessed worst-case influence of OUT-of-region input on the IN-region edit.

    Optimizes a perturbation supported on the *outside* of the region to maximise the
    in-region residual change. Two threat models:
      - ``norm="linf"``: perturbation per-pixel <= eps -> the literal "imperceptible
        out-of-region trigger" magnitude (how much an invisible background tweak steers
        the edit).
      - ``norm="l2"``: perturbation L2 <= eps -> the ratio (in-region change / ||delta||)
        is a *witnessed lower bound* on the editor's L2 cross-region Lipschitz constant
        (must not exceed the certificate -> a soundness check).

    Returns dict: {in_region_change, delta_norm, ratio}. White-box (needs gradients); for
    open-source SOTA editors gradients are available, so this is a sound lower bound on
    their (un)locality.
    """
    src = torch.from_numpy(source_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    outside = (m <= 0.5).float()
    inside = (m > 0.5).float()
    model.eval()
    with torch.no_grad():
        base = model(torch.cat([src, m], dim=1))

    # Random start inside the eps ball (delta=0 is a degenerate point where the L2 gradient
    # vanishes); a smooth squared objective then drives PGD ascent.
    g0 = torch.Generator().manual_seed(seed)
    if norm == "linf":
        delta = ((torch.rand(src.shape, generator=g0) * 2 - 1) * eps * 0.5)
    else:
        d0 = torch.randn(src.shape, generator=g0)
        delta = d0 / (d0.flatten().norm() + 1e-12) * eps * 0.5
    delta = (delta * outside).detach().requires_grad_(True)

    for _ in range(n_steps):
        out = model(torch.cat([(src + delta * outside).clamp(0, 1), m], dim=1))
        obj = ((out - base) * inside).pow(2).sum()
        (grad,) = torch.autograd.grad(obj, delta)
        with torch.no_grad():
            if norm == "linf":
                delta = (delta + step * grad.sign()).clamp(-eps, eps) * outside
            else:  # l2 ball projection on the outside-restricted perturbation
                delta = delta + step * grad / (grad.flatten().norm() + 1e-12)
                delta = delta * outside
                dn = delta.flatten().norm()
                if dn > eps:
                    delta = delta * (eps / dn)
        delta.requires_grad_(True)

    with torch.no_grad():
        d = delta * outside
        out = model(torch.cat([(src + d).clamp(0, 1), m], dim=1))
        change = float(((out - base) * inside).flatten().norm().item())
        dnorm = float(d.abs().max().item()) if norm == "linf" else float(d.flatten().norm().item())
        ratio = change / (dnorm + 1e-12)
    return {"in_region_change": change, "delta_norm": dnorm, "ratio": ratio}


def out_of_region_influence(model: torch.nn.Module, source_rgb: np.ndarray, mask: np.ndarray,
                            n_trials: int = 16, eps: float = 0.1, seed: int = 0) -> float:
    """Largest observed change of the IN-region residual when only OUT-of-region input
    pixels are perturbed -- the empirical 'does the background leak into the edit?' number.

    For our Lipschitz editor this is finite and bounded by the certificate; for a global
    editor it is typically large. ``source_rgb`` HxWx3 uint8, ``mask`` HxW in [0,1].
    """
    device = torch.device("cpu")
    src = torch.from_numpy(source_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    inside = m > 0.5
    outside = ~inside

    g = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    with torch.no_grad():
        base = model(torch.cat([src, m], dim=1))
        worst = 0.0
        for _ in range(n_trials):
            d = torch.randn(src.shape, generator=g)
            d = d * outside.float()                      # perturb only outside the region
            d = eps * d / (d.flatten().norm() + 1e-12)
            pert = model(torch.cat([src + d, m], dim=1))
            change = ((pert - base) * inside.float()).flatten().norm().item()  # inside change
            worst = max(worst, change)
    return worst
