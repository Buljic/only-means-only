"""Tests for Certified Edit Sensitivity: bound soundness + footprint + influence."""

import numpy as np
import torch

from src.editor.residual_editor import ResidualEditor
from src.eval.certify import (
    adversarial_out_of_region_influence,
    certify_sensitivity,
    empirical_sensitivity,
    out_of_region_influence,
    verify_sensitivity_certificate,
)
from src.eval.locality import verify_zero_leakage
from src.models.lipschitz import LipschitzResidualNet, conv_lipschitz_bound


def test_maxmin_is_1lipschitz_and_keeps_certificate_sound():
    import torch

    from src.models.lipschitz import LipschitzResidualNet, MaxMin

    mm = MaxMin()
    x, y = torch.randn(2, 8, 5, 5), torch.randn(2, 8, 5, 5)
    assert (mm(x) - mm(y)).flatten().norm() <= (x - y).flatten().norm() + 1e-5   # 1-Lipschitz

    size = 32
    net = LipschitzResidualNet(hidden=16, n_layers=4, circular=True, act="maxmin")
    cert = certify_sensitivity(net, input_size=size)
    emp = empirical_sensitivity(net, torch.randn(1, 4, size, size), n_trials=24, eps=0.2)
    assert emp <= cert.lipschitz_bound + 1e-3      # certificate still sound with MaxMin


def test_circular_conv_fft_is_sound_and_tighter_than_sum_bound():
    import torch

    from src.models.lipschitz import conv_lipschitz_bound, conv_operator_norm_fft

    n = 24
    conv = torch.nn.Conv2d(3, 4, 3, padding=1, padding_mode="circular")
    b_fft = conv_operator_norm_fft(conv.weight.detach(), n)   # exact op-norm for circular conv
    b_sum = conv_lipschitz_bound(conv.weight.detach())         # loose sum-of-offsets bound

    worst = 0.0
    for _ in range(60):
        d = torch.randn(1, 3, n, n)
        o = conv(d) - conv(torch.zeros_like(d))
        worst = max(worst, o.flatten().norm().item() / d.flatten().norm().item())
    assert worst <= b_fft + 1e-3        # FFT op-norm is sound
    assert b_fft <= b_sum + 1e-6        # and provably tighter than the sum bound


def test_circular_model_certificate_is_tight_and_sound():
    import torch

    from src.models.lipschitz import (
        LipschitzResidualNet,
        is_circular,
        model_certified_lipschitz,
        model_lipschitz_bound,
    )

    net = LipschitzResidualNet(hidden=16, n_layers=4, circular=True)
    assert is_circular(net)
    size = 32
    tight = model_certified_lipschitz(net, size)   # sound (circular) certificate
    loose = model_lipschitz_bound(net)             # sum bound
    assert tight < loose                           # materially tighter

    cert = certify_sensitivity(net, input_size=size)
    assert cert.tight and abs(cert.lipschitz_bound - tight) < 1e-4
    emp = empirical_sensitivity(net, torch.randn(1, 4, size, size), n_trials=24, eps=0.2)
    assert emp <= cert.lipschitz_bound + 1e-3      # soundness vs the tight certificate


def test_conv_bound_is_a_valid_upper_bound():
    """A conv is linear, so observed ||conv(d)||/||d|| must never exceed the bound."""
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(4, 6, 3, padding=1)
    bound = conv_lipschitz_bound(conv.weight.detach())

    worst = 0.0
    for _ in range(30):
        d = torch.randn(1, 4, 24, 24)
        out = conv(d) - conv(torch.zeros_like(d))   # = conv applied to d (minus bias)
        worst = max(worst, out.flatten().norm().item() / d.flatten().norm().item())
    assert worst <= bound + 1e-4
    assert bound > 0


def test_sensitivity_certificate_is_sound():
    torch.manual_seed(0)
    net = LipschitzResidualNet(hidden=16, n_layers=4)
    x = torch.randn(1, 4, 32, 32)

    ok, emp, certified = verify_sensitivity_certificate(net, x, n_trials=24, eps=0.2)
    assert ok                                   # empirical <= certified (soundness)
    assert emp <= certified + 1e-4
    cert = certify_sensitivity(net)
    assert cert.n_conv_layers == 4 and cert.lipschitz_bound == certified


def test_lipschitz_editor_keeps_exact_zero_footprint():
    """The Lipschitz net composes with the certified composite -> zero leakage."""
    rng = np.random.default_rng(0)
    source = rng.integers(40, 200, (48, 48, 3), dtype=np.uint8)
    mask = np.zeros((48, 48), dtype=np.float32)
    mask[12:36, 12:36] = 1.0

    editor = ResidualEditor(LipschitzResidualNet(hidden=16, n_layers=4),
                            feather_radius=3, outside_threshold=0.01)
    edited = editor.edit(source, mask)
    ok, violation = verify_zero_leakage(source, edited, mask, threshold=0.01, feather_radius=3)
    assert ok and violation == 0.0


def test_out_of_region_influence_is_bounded_by_certificate():
    """Perturbing only outside pixels moves the inside edit by a certified-bounded amount."""
    torch.manual_seed(0)
    net = LipschitzResidualNet(hidden=16, n_layers=4)
    rng = np.random.default_rng(0)
    source = rng.integers(40, 200, (32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    eps = 0.2
    influence = out_of_region_influence(net, source, mask, n_trials=16, eps=eps)
    cert = certify_sensitivity(net)
    assert 0.0 <= influence <= cert.max_edit_change(eps) + 1e-4


def test_adversarial_influence_beats_random_and_respects_certificate():
    """PGD finds a stronger out-of-region trigger than random, and the L2 ratio it
    witnesses is a valid lower bound on the editor's certified Lipschitz constant."""
    torch.manual_seed(0)
    net = LipschitzResidualNet(hidden=16, n_layers=4)
    rng = np.random.default_rng(0)
    source = rng.integers(40, 200, (32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    eps = 0.2
    rand = out_of_region_influence(net, source, mask, n_trials=24, eps=eps)
    adv = adversarial_out_of_region_influence(net, source, mask, eps=eps, n_steps=40, norm="l2")
    cert = certify_sensitivity(net)

    assert adv["in_region_change"] >= rand - 1e-6      # PGD >= random
    assert adv["ratio"] <= cert.lipschitz_bound + 1e-3  # witnessed lower bound <= certificate

    # L-infinity threat model returns a finite "imperceptible trigger" magnitude.
    adv_inf = adversarial_out_of_region_influence(net, source, mask, eps=0.03, norm="linf")
    assert adv_inf["in_region_change"] >= 0.0 and adv_inf["delta_norm"] <= 0.03 + 1e-6
