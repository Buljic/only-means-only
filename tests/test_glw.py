"""Tests for the Global-Read, Local-Write editor."""

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from src.eval.locality import audit_editor_sensitivity, verify_zero_leakage
from src.models.glw_editor import GLWEditor, GLWNet, select_gate


def _sample(size=48):
    rng = np.random.default_rng(0)
    source = rng.integers(40, 200, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.float32)
    mask[size // 3:2 * size // 3, size // 3:2 * size // 3] = 1.0
    return source, mask


def test_glw_zero_sensitivity_at_gate0_and_tunable_above():
    torch.manual_seed(0)
    net = GLWNet(hidden=16, ctx_dim=8, n_layers=4, gate=0.0)
    source, mask = _sample()
    halo = 4
    ed = GLWEditor(net, halo=halo)
    region = mask > 0.5
    beyond_W = ~binary_dilation(region, iterations=halo)

    rng = np.random.default_rng(1)
    pert = source.copy()
    pert[beyond_W] = rng.integers(0, 256, (int(beyond_W.sum()), 3), dtype=np.uint8)

    # gate = 0: the in-region edit is provably independent of beyond-window pixels.
    net.gate = 0.0
    np.testing.assert_array_equal(ed.edit(source, mask)[region], ed.edit(pert, mask)[region])

    # gate > 0: the global READ is active (edit responds to beyond-window content) and the
    # out-of-region influence scales with the gate (-> 0 as gate -> 0).
    def influence(g):
        net.gate = g
        a = ed.edit(source, mask)[region].astype(np.float32)
        b = ed.edit(pert, mask)[region].astype(np.float32)
        return float(np.abs(a - b).mean())

    lo, hi = influence(0.05), influence(0.6)
    assert hi > lo > 0.0           # global read is active and gate-controllable

    # footprint preserved regardless of gate
    net.gate = 0.3
    ok, _ = verify_zero_leakage(source, ed.edit(source, mask), mask, threshold=0.01, feather_radius=3)
    assert ok


def test_glw_global_sensitivity_bound_zero_at_gate0_and_linear():
    net = GLWNet(hidden=16, ctx_dim=8, n_layers=4, gate=0.0)
    assert net.global_sensitivity_bound() == 0.0
    net.gate = 0.1
    b1 = net.global_sensitivity_bound()
    net.gate = 0.2
    b2 = net.global_sensitivity_bound()
    assert b1 > 0 and abs(b2 - 2 * b1) < 1e-4      # the bound is exactly linear in the gate


def test_select_gate_returns_smallest_passing_gate():
    torch.manual_seed(0)
    net = GLWNet(hidden=16, ctx_dim=8, n_layers=4, gate=0.0)
    source, mask = _sample()
    ed = GLWEditor(net, halo=4)
    region = mask > 0.5

    def quality_fn(edited):
        return float(np.abs(edited[region].astype(np.float32) - source[region].astype(np.float32)).mean()) / 50.0

    gates = (0.0, 0.1, 0.2, 0.4)
    thr = 0.05
    quals = {}
    for g in gates:
        net.gate = g
        quals[g] = quality_fn(ed.edit(source, mask))
    expected = next((g for g in sorted(gates) if quals[g] >= thr), max(gates))

    res = select_gate(net, ed, source, mask, quality_fn, gates=gates, threshold=thr)
    assert res["gate"] == expected
    assert abs(res["certificate"] - 2 * expected * (3 ** 0.5)) < 1e-6


def test_glw_tight_bound_is_sound():
    """No out-of-window perturbation can move the in-region edit (per-pixel L2, normalized)
    beyond the certified bound."""
    torch.manual_seed(0)
    net = GLWNet(hidden=16, ctx_dim=8, n_layers=4, gate=0.3)
    source, mask = _sample()
    halo = 4
    ed = GLWEditor(net, halo=halo)
    region = mask > 0.5
    beyond_W = ~binary_dilation(region, iterations=halo)

    bound = net.global_sensitivity_bound()                     # normalized per-pixel L2
    e0 = ed.edit(source, mask).astype(np.float32) / 255.0
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(8):
        pert = source.copy()
        pert[beyond_W] = rng.integers(0, 256, (int(beyond_W.sum()), 3), dtype=np.uint8)
        e1 = ed.edit(pert, mask).astype(np.float32) / 255.0
        dpp = np.sqrt(((e1 - e0) ** 2).sum(axis=2))[region]    # per-pixel L2 over channels
        worst = max(worst, float(dpp.max()))
    assert worst <= bound + 2.0 / 255.0                        # sound (+ quantization slack)


def test_glw_cross_region_certificate_is_sound_and_tighter():
    """The threat-model-aware cross-region certificate must (1) upper-bound the empirical
    adversarial trigger, and (2) be strictly tighter than the absolute gated bound for an
    imperceptible epsilon -- the non-vacuous certificate that Lipschitz training will shrink."""
    torch.manual_seed(0)
    net = GLWNet(hidden=16, ctx_dim=8, n_layers=4, gate=0.1)
    source, mask = _sample(48)
    ed = GLWEditor(net, halo=8)
    fn = lambda s, m, k: ed.edit(s, m)
    region = mask > 0.5
    omega = ~binary_dilation(region, iterations=ed.halo)
    n_region = int(region.sum())
    n_outside = int(omega.sum())
    eps = 8.0

    a = audit_editor_sensitivity(
        fn, source, mask, 0.5, eps=eps, n_random=4, spsa_steps=6,
        perturb_mask=omega,
    )
    cert255 = net.certified_influence(eps / 255.0, n_region, n_outside, input_size=48) * 255.0
    absolute = net.global_sensitivity_bound() * 255.0

    assert net.cross_region_lipschitz(48) > 0.0
    assert a["adv_influence"] <= cert255 + 1e-6        # SOUND: empirical never exceeds the cert
    assert cert255 < absolute                          # TIGHTER than the old vacuous bound
    assert net.certified_influence(eps / 255.0, n_region, 0, input_size=48) == 0.0

    # The cross-region Lipschitz scales linearly with the gate (-> 0 as gate -> 0).
    net.gate = 0.2
    l2 = net.cross_region_lipschitz(48)
    net.gate = 0.1
    l1 = net.cross_region_lipschitz(48)
    assert abs(l2 - 2 * l1) < 1e-4
