"""Lipschitz-bounded editor + a *sound* convolutional Lipschitz bound.

This is the mechanism behind the "Certified Edit Sensitivity" contribution: an editor
whose response to out-of-region content is provably bounded. We use a plain stride-1
convolutional stack (no pooling/upsampling) so a valid global Lipschitz constant follows
from per-layer bounds and 1-Lipschitz ReLUs.

Sound per-conv bound. A stride-1 conv with kernel ``K[out, in, kh, kw]`` equals
``sum_{i,j} S_{ij} . (1x1 conv with matrix K[:, :, i, j])`` where each ``S_{ij}`` is a
(zero-padded) spatial shift with operator 2-norm <= 1. By the triangle inequality the
conv's operator 2-norm is therefore upper-bounded by

    L_conv <= sum_{i,j} sigma_max( K[:, :, i, j] )                      (Eq. *)

i.e. the sum over kernel offsets of the spectral norm of the per-offset channel matrix.
This is a *valid* upper bound (unlike the common reshaped-matrix spectral norm), which is
what a certificate requires. A tighter conv certificate (Sedghi et al. / Singla & Feizi)
is a future rigor upgrade.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_lipschitz_bound(weight: torch.Tensor) -> float:
    """Sound upper bound on a stride-1 conv's Lipschitz (2-norm), via Eq. (*)."""
    out_ch, in_ch, kh, kw = weight.shape
    total = 0.0
    for i in range(kh):
        for j in range(kw):
            m = weight[:, :, i, j]                       # [out, in]
            total += float(torch.linalg.matrix_norm(m, ord=2).item())
    return total


def model_lipschitz_bound(model: nn.Module) -> float:
    """Product of per-conv bounds (ReLU is 1-Lipschitz) -> sound global bound."""
    bound = 1.0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            bound *= conv_lipschitz_bound(m.weight.detach())
    return bound


@torch.no_grad()
def conv_operator_norm_fft(weight: torch.Tensor, n: int) -> float:
    """Tight conv operator 2-norm via the FFT method (Sedghi et al., 2019).

    For circular convolution the singular values of the conv operator are the singular
    values of the per-frequency transfer matrices ``FFT2(K_padded)[u, v] in C^{out x in}``;
    the operator 2-norm is their max. Exact for circular padding, a close estimate of the
    *true* Lipschitz constant for zero-padded convs at usual resolutions. Used to probe how
    far the loose ``conv_lipschitz_bound`` overestimates -- i.e. whether a *useful*
    certificate is reachable.
    """
    out_ch, in_ch, kh, kw = weight.shape
    if int(n) < max(kh, kw):
        raise ValueError(
            f"conv_operator_norm_fft: grid n={n} is smaller than the kernel ({kh}x{kw}); "
            "the circular transfer matrices would alias and the norm would not be sound"
        )
    padded = torch.zeros(out_ch, in_ch, n, n, dtype=weight.dtype, device=weight.device)
    padded[:, :, :kh, :kw] = weight
    transfer = torch.fft.fft2(padded, dim=(-2, -1)).permute(2, 3, 0, 1)  # [n,n,out,in] complex
    svals = torch.linalg.matrix_norm(transfer, ord=2)                    # [n,n] (largest s.v.)
    return float(svals.abs().max().item())


@torch.no_grad()
def model_lipschitz_bound_fft(model: nn.Module, input_size: int) -> float:
    """Product of per-conv FFT operator norms. EXACT per layer for *circular* convolution
    (Sedghi et al., 2019), so for a circular-padded stack this product is a SOUND global
    Lipschitz certificate (sub-multiplicativity); for zero-padded convs it is an estimate.

    RESOLUTION SCOPE. The singular values are those of the conv operator on an
    ``input_size x input_size`` circular grid, so the returned norm is exact *at that resolution
    and no other*; it is not resolution-generic. Monotonicity in the grid size does not hold in
    general: the frequency grids at different sizes are not nested, so they can miss different
    maxima. A 3x3 kernel whose only nonzero row is [1, -1, 0] has circular operator norm 2 at
    n=4 but 2*cos(pi/10) < 2 at n=5. Certifying at a large fixed grid is not a viable fix --
    the transfer tensor is O(out * in * n^2) and reaches 5.4 GB at n=1024 for the real
    global_synth shape. Callers must therefore certify at the resolution they evaluate at, and
    record it (the frontier artifacts carry ``input_size``).
    """
    bound = 1.0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            bound *= conv_operator_norm_fft(m.weight.detach(), input_size)
    return bound


def is_circular(model: nn.Module) -> bool:
    """True iff every Conv2d uses circular padding (so the FFT bound is a sound certificate)."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return len(convs) > 0 and all(m.padding_mode == "circular" for m in convs)


def model_certified_lipschitz(model: nn.Module, input_size: int | None = None) -> float:
    """SOUND global Lipschitz certificate.

    Uses the *tight* FFT operator norm when the network is circular-padded (then the FFT
    per-layer norm is exact, Sedghi et al.), otherwise falls back to the (loose but sound
    for any padding) sum-of-offsets bound. This is the fix for "the certificate is loose":
    a circular-padded certified editor certifies the small true Lipschitz, not the ~10-50x
    overestimate of the sum bound.
    """
    unsupported = [
        type(module).__name__
        for module in model.modules()
        if not isinstance(module, nn.Conv2d)
        and any(True for _ in module.parameters(recurse=False))
    ]
    if unsupported:
        names = ", ".join(sorted(set(unsupported)))
        raise TypeError(
            "No sound certificate is implemented for parameterized module(s): "
            f"{names}. Only Conv2d stacks with 1-Lipschitz parameter-free activations "
            "are currently certifiable."
        )
    if input_size is not None and is_circular(model):
        return model_lipschitz_bound_fft(model, input_size)
    return model_lipschitz_bound(model)


class MaxMin(nn.Module):
    """GroupSort-2 (MaxMin) activation: 1-Lipschitz AND gradient-norm-preserving, so it is a
    far better Lipschitz function approximator than ReLU (Anil et al., 2019) -- more edit
    quality per unit of certified Lipschitz. Channel count is preserved (needs even C)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        a = x.view(b, c // 2, 2, h, w)
        mx = a.max(dim=2).values
        mn = a.min(dim=2).values
        return torch.cat([mx, mn], dim=1)


def _activation(name: str) -> nn.Module:
    return MaxMin() if name == "maxmin" else nn.ReLU(inplace=True)


class LipschitzResidualNet(nn.Module):
    """Plain stride-1 conv stack: concat(source, mask) [B,4,H,W] -> RGB residual [B,3,H,W].

    No pooling/upsampling/normalisation so ``model_lipschitz_bound`` is a sound global
    Lipschitz constant. Less expressive than a U-Net by design -- the trade-off (a
    certifiable editor) is exactly the point.
    """

    def __init__(self, in_ch: int = 4, out_ch: int = 3, hidden: int = 32, n_layers: int = 4,
                 circular: bool = True, act: str = "maxmin"):
        super().__init__()
        assert n_layers >= 2
        assert act != "maxmin" or hidden % 2 == 0, "MaxMin needs an even hidden width"
        # Circular padding makes the FFT operator norm an EXACT, SOUND per-layer Lipschitz
        # (Sedghi et al.), so the certificate is tight by construction. MaxMin (1-Lipschitz,
        # norm-preserving) gives more expressivity per unit Lipschitz than ReLU.
        pm = "circular" if circular else "zeros"
        def conv(i, o):
            return nn.Conv2d(i, o, 3, padding=1, padding_mode=pm)
        layers: list[nn.Module] = [conv(in_ch, hidden)]
        for _ in range(n_layers - 2):
            layers += [_activation(act), conv(hidden, hidden)]
        layers += [_activation(act), conv(hidden, out_ch)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def lipschitz_bound(self) -> float:
        """Certified upper bound on this network's input->output 2-norm sensitivity."""
        return model_lipschitz_bound(self)


class LipschitzPointwiseNet(nn.Module):
    """1x1-convolutional context map with a sound Euclidean Lipschitz certificate.

    This is used for fixed, spatially collapsed declassification statistics. Keeping the
    network pointwise avoids pretending that a learned spatial encoder determines which
    information is "physical" or "semantic"; the only admissible information is fixed by
    the statistic computed before this module.
    """

    def __init__(self, in_ch: int, out_ch: int, hidden: int = 16, n_layers: int = 3,
                 act: str = "maxmin"):
        super().__init__()
        assert n_layers >= 2
        assert act != "maxmin" or hidden % 2 == 0, "MaxMin needs an even hidden width"
        layers: list[nn.Module] = [nn.Conv2d(in_ch, hidden, 1)]
        for _ in range(n_layers - 2):
            layers += [_activation(act), nn.Conv2d(hidden, hidden, 1)]
        layers += [_activation(act), nn.Conv2d(hidden, out_ch, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
