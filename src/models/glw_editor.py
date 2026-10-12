"""Global-Read, Local-Write (GLW) editor.

Resolves the central tension (high-quality edits need global context, but unrestricted
global context leaks: the background secretly steers the foreground). The model READS the
whole image for context, but the WRITE depends on:

  - the masked local window (region + halo): UNRESTRICTED, but provably has zero influence
    from pixels outside the window (input masking), and
  - a single Lipschitz-bounded, *gated* global-context vector g (a pooled global summary).

The gate ``gamma`` is the knob: ``gamma = 0`` => the write is a pure function of the local window
=> provably zero out-of-region sensitivity (recovers the input-masking wrapper); ``gamma > 0``
=> the global channel restores quality at a *tunable, bounded* out-of-region sensitivity
that scales with ``gamma`` (and vanishes as ``gamma -> 0``). Footprint is preserved by
``composite_local``. A tight closed-form certificate on the global channel (the broadcast-g
Lipschitz) is the remaining research; the *architecture* gives the guarantee structurally.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.editor.cpu_editor import composite_local
from src.models.lipschitz import LipschitzPointwiseNet, LipschitzResidualNet


class GLWBase(nn.Module):
    """Shared Global-Read, Local-Write forward + certificate. Subclasses provide the synth
    modules ``local_synth`` (in 3+1 -> 3), ``global_enc`` (in 3 -> ctx) and ``global_synth``
    (in 3+1+ctx -> 3), and may accept an optional attention ``window`` mask via ``_run``."""

    ctx_dim: int
    gate: float

    def _local(self, x, window=None):
        return self.local_synth(x)

    def _global_enc(self, x):
        return self.global_enc(x)

    def _global(self, x, window=None):
        return self.global_synth(x)

    def forward(self, masked_local: torch.Tensor, mask: torch.Tensor, full_image: torch.Tensor,
                window: torch.Tensor | None = None) -> torch.Tensor:
        local_res = self._local(torch.cat([masked_local, mask], dim=1), window)
        g = torch.tanh(self._global_enc(full_image).mean(dim=(2, 3), keepdim=True))  # bounded global READ
        g_b = g.expand(-1, -1, masked_local.shape[2], masked_local.shape[3])
        # The global residual is MAGNITUDE-bounded by tanh, so the out-of-window channel can
        # shift each edited pixel by at most +/- gate per channel -- a tight, depth-independent
        # certificate (no exploding Lipschitz product). out-of-window enters ONLY here.
        global_res = torch.tanh(self._global(torch.cat([masked_local, mask, g_b], dim=1), window))
        return local_res + self.gate * global_res

    def global_sensitivity_bound(self, input_size: int | None = None) -> float:
        """TIGHT, depth-independent bound on out-of-window -> in-region edit (per-pixel L2,
        normalized [0,1]).

        Derivation. Out-of-window pixels reach the write ONLY through the global term
        ``gate * tanh(global_synth(...))``; the masked local path has zero out-of-window
        influence. Because ``tanh`` bounds ``global_res`` to ``[-1,1]`` per channel, the
        global term lies in ``[-gate, gate]`` per channel regardless of the network's weights,
        so between any two inputs the per-pixel change of the global term is at most
        ``2*gate`` per channel, i.e. ``2*gate*sqrt(3)`` in per-pixel L2 over the 3 channels.
        ``clamp`` (1-Lipschitz) and the mask-weighted composite (weight <= 1) only shrink it.
        The bound is exact in the worst case, 0 at gate=0, and linear in the gate -- no
        exploding Lipschitz product. ``input_size`` is unused (kept for interface symmetry).
        """
        return 2.0 * self.gate * (3.0 ** 0.5)

    def cross_region_lipschitz(self, input_size: int | None = None) -> float:
        """SOUND, training-responsive Lipschitz of (out-of-region input) -> (in-region output).

        Unlike ``global_sensitivity_bound`` (an *absolute* worst-case diameter, vacuous for the
        imperceptible-trigger threat model and constant w.r.t. the weights), this is a true
        *Lipschitz* constant: ``|| (E(I+d) - E(I))_R ||_2 <= L_cross * ||d||_2`` for any d on the
        background. Out-of-region pixels reach the write ONLY through
        ``g = tanh(mean_spatial(global_enc(full_image)))`` and then ``global_synth(.., g)``;
        tanh and mean-pool are norm-non-increasing, and the mean-pool's 1/sqrt(N) shrink is
        exactly cancelled by broadcasting g back to N positions, leaving

            L_cross = gate * L(global_enc) * L(global_synth)

        with each L the *certified* (FFT-exact for circular) sub-network Lipschitz. The masked
        local path contributes 0 (its out-of-window pixels are a constant fill). Because L_cross
        is the product of the sub-network Lipschitz constants, **Lipschitz-parameterized training
        drives it down** -- this is the knob that turns the vacuous bound into a useful one.
        """
        from src.models.lipschitz import model_certified_lipschitz
        l_enc = model_certified_lipschitz(self.global_enc, input_size)
        l_syn = model_certified_lipschitz(self.global_synth, input_size)
        return float(self.gate * l_enc * l_syn)

    def certified_influence(self, eps01: float, n_region: int, n_outside: int,
                            input_size: int | None = None) -> float:
        """Sound bound on the audit's in-region per-pixel RMS change (normalized [0,1]) under an
        L-inf ``Omega`` perturbation of size ``eps01``, in the SAME units as the empirical
        ``adv_trigger``. Returns the tighter of the two sound bounds:

          - absolute gated bound  ``2*gate*sqrt(3)``  (input-independent), and
          - Lipschitz bound  ``L_cross * eps01 * sqrt(|Omega| / |R|)``  (threat-model aware:
            ||d||_2 <= eps01*sqrt(3|Omega|) for an L-inf ball; RMS over R divides by sqrt(3|R|)).

        For imperceptible ``eps01`` and a trained (small-Lipschitz) editor the second is far
        smaller -- the non-vacuous certificate. ``min`` keeps soundness (both are valid).
        """
        import math
        if n_region <= 0 or n_outside <= 0 or eps01 <= 0:
            return 0.0
        # Units: `adv_trigger` (locality.audit_editor_sensitivity) measures MEAN ABSOLUTE in-region
        # change per pixel-channel, not the per-pixel L2 that `global_sensitivity_bound` bounds.
        # Each channel of the gated term moves by at most 2*gate, so the mean-absolute bound is
        # 2*gate -- the sqrt(3) that belongs in the per-pixel-L2 diameter is pure slack here.
        # `global_sensitivity_bound` itself is UNCHANGED: it is correct in its own unit and is
        # tested as such (tests/test_glw.py::test_glw_..._bound_is_sound).
        absolute = 2.0 * abs(float(self.gate))
        ratio = math.sqrt(float(n_outside) / float(n_region))
        lip = self.cross_region_lipschitz(input_size) * float(eps01) * ratio
        return float(min(absolute, lip))


class GLWNet(GLWBase):
    """Convolutional GLW: circular-conv MaxMin synths (cheap, deterministic)."""

    def __init__(self, hidden: int = 32, ctx_dim: int = 16, n_layers: int = 4, gate: float = 0.1):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.gate = float(gate)
        self.global_enc = LipschitzResidualNet(3, ctx_dim, hidden, 3, circular=True, act="maxmin")
        self.local_synth = LipschitzResidualNet(3 + 1, 3, hidden, n_layers, circular=True, act="maxmin")
        self.global_synth = LipschitzResidualNet(3 + 1 + ctx_dim, 3, hidden, n_layers, circular=True, act="maxmin")


class AffineContextSynth(nn.Module):
    """Exact affine-in-context physical write head.

    The local window and mask produce concrete coefficient maps ``A`` and ``b``. The admitted
    global context enters only as ``b + sum_j A_j z_j`` before the outer tanh. For a fixed local
    window this is affine in ``z`` exactly, so its reachable interval over a context box is attained
    at an explicit box corner. The parameter head is tanh-bounded to keep coefficient magnitudes
    controlled during the first retraining pilot.
    """

    def __init__(self, ctx_dim: int, hidden: int = 32, n_layers: int = 4):
        super().__init__()
        self.ctx_dim = int(ctx_dim)
        self.param_net = LipschitzResidualNet(
            4, 3 * (self.ctx_dim + 1), hidden, n_layers, circular=True, act="maxmin"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_mask = x[:, :4]
        params = torch.tanh(self.param_net(local_mask))
        b = params[:, :3]
        a = params[:, 3:].view(x.shape[0], 3, self.ctx_dim, x.shape[2], x.shape[3])
        z = x[:, 4:4 + self.ctx_dim]
        return b + (a * z.unsqueeze(1)).sum(dim=2)


class TypedGLWNet(GLWBase):
    """Declassification-aware Global-Read, Local-Write editor.

    The two global paths have different *formal* contracts:

    * ``declassified`` (legacy field name: ``physical``) receives only the fixed statistic
      ``P(x) = [channel mean, channel standard deviation]``. The guarantee is therefore
      explicit: when two images agree on the trusted window and have the same ``P(x)``, this
      path produces exactly the same result. The statistic is useful for coarse color and
      illumination adaptation, but it is not claimed to be intrinsically "semantic-free".
    * ``semantic`` receives the unrestricted full image and is separately gated/certified.

    This is a real information-flow boundary: a learned encoder can no longer smuggle arbitrary
    spatial content through a branch merely because that branch was named ``physical``.
    """

    DECLASSIFICATION_SCHEMA = "rgb_mean_std_v1"
    DECLASSIFIED_INPUT_CHANNELS = 6

    def __init__(
        self,
        hidden: int = 32,
        ctx_dim_physical: int = 4,
        ctx_dim_semantic: int = 16,
        n_layers: int = 4,
        gamma_physical: float = 0.1,
        gamma_semantic: float = 0.02,
        phys_downsample: int = 8,
        physical_encoder: str = "statistics",
        physical_interface: str = "nonlinear",
    ):
        super().__init__()
        self.ctx_dim = int(ctx_dim_physical) + int(ctx_dim_semantic)
        self.ctx_dim_physical = int(ctx_dim_physical)
        self.ctx_dim_semantic = int(ctx_dim_semantic)
        self.gamma_physical = float(gamma_physical)
        self.gamma_semantic = float(gamma_semantic)
        # Kept in the constructor/checkpoint schema for backward CLI compatibility. The former
        # learned low-pass path was not a semantic type guarantee and is intentionally unused.
        self.phys_downsample = max(1, int(phys_downsample))
        # Compatibility with GLWBase utilities that expect a single gate.
        self.gate = float(max(abs(self.gamma_physical), abs(self.gamma_semantic)))

        self.local_synth = LipschitzResidualNet(3 + 1, 3, hidden, n_layers, circular=True, act="maxmin")
        # ``learned`` is accepted as a checkpoint/CLI compatibility spelling, but it now means a
        # learned pointwise map *after* the fixed P(x) boundary.  It never restores the retired
        # learned encoder over raw/downsampled pixels.
        requested_encoder = str(physical_encoder).lower()
        if requested_encoder == "learned":
            requested_encoder = "statistics"
        if requested_encoder not in {"statistics", "fixed", "affine", "mean_affine", "mean_linear"}:
            raise ValueError("physical_encoder must be one of: statistics, fixed, affine, mean_affine, mean_linear, learned")
        self.physical_encoder_kind = requested_encoder
        requested_interface = str(physical_interface).lower()
        if requested_interface not in {"nonlinear", "affine_context"}:
            raise ValueError("physical_interface must be nonlinear or affine_context")
        self.physical_interface_kind = requested_interface
        if requested_encoder == "fixed":
            from src.models.physical_featurizer import FixedPhysicalEncoder

            self.physical_enc = FixedPhysicalEncoder(self.ctx_dim_physical)
        elif requested_encoder == "affine":
            self.physical_enc = nn.Conv2d(
                self.DECLASSIFIED_INPUT_CHANNELS, self.ctx_dim_physical, kernel_size=1
            )
        elif requested_encoder == "mean_affine":
            self.physical_enc = nn.Conv2d(3, self.ctx_dim_physical, kernel_size=1)
        elif requested_encoder == "mean_linear":
            self.physical_enc = nn.Conv2d(3, self.ctx_dim_physical, kernel_size=1)
        else:
            self.physical_enc = LipschitzPointwiseNet(
                self.DECLASSIFIED_INPUT_CHANNELS, self.ctx_dim_physical, hidden=max(2, hidden),
                n_layers=3, act="maxmin"
            )
        if requested_interface == "affine_context":
            self.physical_synth = AffineContextSynth(self.ctx_dim_physical, hidden, n_layers)
        else:
            self.physical_synth = LipschitzResidualNet(
                3 + 1 + self.ctx_dim_physical, 3, hidden, n_layers, circular=True, act="maxmin"
            )
        self.semantic_enc = LipschitzResidualNet(3, self.ctx_dim_semantic, hidden, 3, circular=True, act="maxmin")
        self.semantic_synth = LipschitzResidualNet(
            3 + 1 + self.ctx_dim_semantic, 3, hidden, n_layers, circular=True, act="maxmin"
        )

    @staticmethod
    def declassified_statistics(full_image: torch.Tensor) -> torch.Tensor:
        """Return the fixed admissible global statistic ``P(x)`` as ``[B,6,1,1]``.

        For each RGB channel, mean has L2 Lipschitz constant ``1/sqrt(N)``. Centered RMS
        (standard deviation) is ``||C x||_2/sqrt(N)`` with centering projection ``||C||=1``,
        so it has the same constant. Concatenating the two gives
        ``Lip(P) <= sqrt(2/N)``. This proof covers arbitrary HxW dimensions and replaces the
        fragile downsample/broadcast cancellation used by the legacy learned low-pass path.
        """
        if full_image.ndim != 4 or full_image.shape[1] != 3:
            raise ValueError("declassified_statistics expects [B,3,H,W]")
        mean = full_image.mean(dim=(2, 3), keepdim=True)
        centered = full_image - mean
        std = torch.sqrt(centered.square().mean(dim=(2, 3), keepdim=True) + 1e-12)
        return torch.cat([mean, std], dim=1)

    def _physical_input(self, full_image: torch.Tensor) -> torch.Tensor:
        """Return the formally admitted input for the selected declassification path."""
        if self.physical_encoder_kind == "fixed":
            return full_image
        stats = self.declassified_statistics(full_image)
        return stats[:, :3] if self.physical_encoder_kind in {"mean_affine", "mean_linear"} else stats

    def _broadcast_context(self, ctx: torch.Tensor, h: int, w: int) -> torch.Tensor:
        context = ctx.mean(dim=(2, 3), keepdim=True)
        if self.physical_encoder_kind != "mean_linear":
            context = torch.tanh(context)
        return context.expand(-1, -1, h, w)

    def _physical_context(self, full_image: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Compute the physical/declassified context through the declared information boundary."""
        return self._broadcast_context(self.physical_enc(self._physical_input(full_image)), h, w)

    def forward(
        self,
        masked_local: torch.Tensor,
        mask: torch.Tensor,
        full_image: torch.Tensor,
        window: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_res = self._local(torch.cat([masked_local, mask], dim=1), window)
        h, w = masked_local.shape[-2:]

        gp = self._physical_context(full_image, h, w)
        gs = self._broadcast_context(self.semantic_enc(full_image), h, w)
        physical_res = torch.tanh(self.physical_synth(torch.cat([masked_local, mask, gp], dim=1)))
        semantic_res = torch.tanh(self.semantic_synth(torch.cat([masked_local, mask, gs], dim=1)))
        return local_res + self.gamma_physical * physical_res + self.gamma_semantic * semantic_res

    def global_sensitivity_bound(self, input_size: int | None = None) -> float:
        return float(2.0 * (abs(self.gamma_physical) + abs(self.gamma_semantic)) * (3.0 ** 0.5))

    def cross_region_lipschitz_physical(self, input_size: int | None = None) -> float:
        """Sound bound for the explicitly declassified channel.

        For ``statistics``, ``P`` contributes ``sqrt(2/N)`` and broadcasting contributes
        ``sqrt(N)``, yielding ``sqrt(2)`` at every resolution.  For ``fixed``, the fixed RGB/luma
        projection is applied pointwise before spatial averaging; averaging and broadcasting
        cancel exactly, leaving its closed-form projection norm.  Tanh is 1-Lipschitz.
        """
        from src.models.lipschitz import model_certified_lipschitz

        if self.physical_encoder_kind == "fixed":
            l_enc = self.physical_enc.lipschitz()
            declassification_factor = 1.0
        else:
            l_enc = model_certified_lipschitz(self.physical_enc, 1)
            declassification_factor = 2.0 ** 0.5
        if self.physical_interface_kind == "affine_context":
            # The affine head has |A[c,j]| <= 1 by construction. Its per-pixel RGB operator
            # norm from z is therefore bounded by sqrt(3*k); no nonlinear synth Lipschitz product
            # is involved in the admitted global path.
            return float(
                abs(self.gamma_physical) * declassification_factor * l_enc
                * (3.0 * float(self.ctx_dim_physical)) ** 0.5
            )
        l_syn = model_certified_lipschitz(self.physical_synth, input_size)
        return float(abs(self.gamma_physical) * declassification_factor * l_enc * l_syn)

    def cross_region_lipschitz_declassified(self, input_size: int | None = None) -> float:
        """Preferred name for :meth:`cross_region_lipschitz_physical`."""
        return self.cross_region_lipschitz_physical(input_size)

    def cross_region_lipschitz_semantic(self, input_size: int | None = None) -> float:
        from src.models.lipschitz import model_certified_lipschitz

        l_enc = model_certified_lipschitz(self.semantic_enc, input_size)
        l_syn = model_certified_lipschitz(self.semantic_synth, input_size)
        return float(abs(self.gamma_semantic) * l_enc * l_syn)

    def cross_region_lipschitz(self, input_size: int | None = None) -> float:
        return float(
            self.cross_region_lipschitz_physical(input_size)
            + self.cross_region_lipschitz_semantic(input_size)
        )

    def certified_influence_typed(
        self,
        eps01: float,
        n_region: int,
        n_outside: int,
        input_size: int | None = None,
    ) -> dict[str, float]:
        """Per-channel sound influence bounds in normalized [0,1] RMS units."""
        import math

        if n_region <= 0 or n_outside <= 0 or eps01 <= 0:
            return {"declassified": 0.0, "physical": 0.0, "semantic": 0.0, "total": 0.0}
        ratio = math.sqrt(float(n_outside) / float(n_region))

        def _channel(gamma: float, lip: float) -> float:
            # Units: mean-absolute (see GLWBase.certified_influence) -> 2*gamma, not 2*gamma*sqrt(3).
            absolute = 2.0 * abs(float(gamma))
            local = float(lip) * float(eps01) * ratio
            return float(min(absolute, local))

        physical = _channel(self.gamma_physical, self.cross_region_lipschitz_physical(input_size))
        semantic = _channel(self.gamma_semantic, self.cross_region_lipschitz_semantic(input_size))
        return {
            "declassified": physical,
            "physical": physical,  # compatibility alias for existing artifacts/CLI columns
            "semantic": semantic,
            "total": physical + semantic,
        }

    def certified_influence(
        self,
        eps01: float,
        n_region: int,
        n_outside: int,
        input_size: int | None = None,
    ) -> float:
        return float(self.certified_influence_typed(eps01, n_region, n_outside, input_size)["total"])


def select_gate(net: "GLWBase", editor: "GLWEditor", source: np.ndarray, mask: np.ndarray,
                quality_fn, gates=(0.0, 0.05, 0.1, 0.2, 0.4), threshold: float = 0.5) -> dict:
    """Adaptive certificate-quality control: pick the SMALLEST gate (tightest certificate /
    smallest epsilon) whose edit still passes ``quality_fn(edited) >= threshold``.

    This operationalizes the locality<->context<->quality trade-off: small gate = stronger
    certificate, larger gate = richer global context. ``quality_fn`` is pluggable -- at run
    time it is an identity-preservation + prompt-fidelity check. Returns the chosen gate, its
    certificate, the quality, and whether the threshold was met (else the largest gate).
    """
    last = None
    for g in sorted(gates):
        net.gate = float(g)
        q = float(quality_fn(editor.edit(source, mask)))
        rec = {"gate": float(g), "certificate": net.global_sensitivity_bound(), "quality": q,
               "passed": q >= threshold}
        if rec["passed"]:
            return rec
        last = rec
    return last  # none passed -> richest-context gate


class GLWEditor:
    """Inference wrapper: global-read, local-write edit with hard-preserve footprint."""

    def __init__(self, net: GLWBase, halo: int = 8, fill: float | None = None,
                 feather_radius: int = 3, outside_threshold: float = 0.01):
        self.net = net.eval()
        self.halo = halo
        self.fill = fill
        self.feather_radius = feather_radius
        self.outside_threshold = outside_threshold

    @torch.no_grad()
    def edit(self, source_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from src.utils.masks import dilate_window

        # dilate_window, never raw binary_dilation: scipy treats iterations=0 as "dilate to
        # convergence", which at halo=0 would hand this supposedly-LOCAL branch the whole image.
        W = dilate_window(mask, self.halo)
        src = torch.from_numpy(source_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        masked = src.clone()
        fill_val = (torch.tensor(self.fill).repeat(3) if self.fill is not None
                    else src[0, :, torch.from_numpy(W)].mean(dim=1))      # depends only on W
        masked[0, :, torch.from_numpy(~W)] = fill_val[:, None]

        residual = self.net(masked, m, src)                              # full image = global READ
        edited = (src + residual).clamp(0, 1)
        edited_rgb = (edited[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return composite_local(source_rgb, edited_rgb, mask,
                               feather_radius=self.feather_radius, outside_threshold=self.outside_threshold)


class TypedGLWEditor(GLWEditor):
    """Inference wrapper for :class:`TypedGLWNet`.

    Kept as a separate class so training/evaluation code can type-dispatch on the new
    editor while reusing the proven GLW local-window preparation and compositing path.
    """
    pass
