"""Tests for the certified-locality toolkit."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.editor.cpu_editor import feather_mask
from src.eval.locality import (
    LocalityCertificate,
    audit_editor_sensitivity,
    compare_editors,
    leakage_report,
    make_certified_local_wrapper,
    make_cpu_editor_fn,
    make_global_context_editor,
    make_global_noise_editor,
    make_global_relight_editor,
    verify_zero_leakage,
)

EDITOR_CFG = SimpleNamespace(feather_radius=3, max_boost=50.0, outside_threshold=0.01)


def _sample():
    rng = np.random.default_rng(0)
    source = rng.integers(40, 210, (96, 96, 3), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.float32)
    mask[32:64, 32:64] = 1.0  # centered square -> a clear outside region
    return source, mask


@pytest.mark.parametrize("mode", ["additive", "divisive"])
def test_certified_zero_leakage_holds_for_ours(mode):
    source, mask = _sample()
    edited = make_cpu_editor_fn(EDITOR_CFG, mode)(source, mask, 0.5)
    ok, violation = verify_zero_leakage(
        source, edited, mask, EDITOR_CFG.outside_threshold, EDITOR_CFG.feather_radius
    )
    assert ok and violation == 0.0


def test_leaky_editor_fails_certificate_and_leaks_globally():
    source, mask = _sample()
    edited = make_global_relight_editor()(source, mask, 0.5)

    ok, violation = verify_zero_leakage(
        source, edited, mask, EDITOR_CFG.outside_threshold, EDITOR_CFG.feather_radius
    )
    assert not ok and violation > 0.0

    rep = leakage_report(source, edited, mask, threshold=0.5)
    assert rep["support_frac"] > 0.9      # changed almost everywhere outside
    assert rep["linf"] > 0.0


def test_leakage_report_zero_on_identity():
    source, mask = _sample()
    rep = leakage_report(source, source.copy(), mask, threshold=0.5)
    assert rep["l1"] == 0.0 and rep["linf"] == 0.0 and rep["support_frac"] == 0.0


def test_certificate_matches_feathered_support():
    _, mask = _sample()
    cert = LocalityCertificate(threshold=0.01, feather_radius=3)
    expected = feather_mask(mask, 3) < 0.01
    np.testing.assert_array_equal(cert.guaranteed_unchanged(mask), expected)
    # The certificate must guarantee a non-trivial fraction of the image.
    assert 0.0 < cert.certified_fraction(mask) < 1.0


def test_sensitivity_audit_distinguishes_global_from_local_editor():
    """The black-box audit must flag a sensitivity-violating (global-context) editor and
    clear a sensitivity-local (per-pixel parametric) one."""
    source, mask = _sample()

    # Parametric additive editor: in-region edit depends only on in-region pixels -> ~0.
    local_fn = make_cpu_editor_fn(EDITOR_CFG, "additive")
    a_local = audit_editor_sensitivity(local_fn, source, mask, eps=8.0, n_random=8, spsa_steps=8)

    # Global-context editor: in-region edit pulls toward the OUTSIDE mean -> large.
    global_fn = make_global_context_editor(pull=0.6)
    a_global = audit_editor_sensitivity(global_fn, source, mask, eps=8.0, n_random=8, spsa_steps=8)

    assert a_global["adv_influence"] >= a_global["random_influence"]      # PGD/SPSA >= random
    assert a_global["adv_influence"] > 1.0                                # background steers the edit
    assert a_local["adv_influence"] < 0.5                                 # parametric editor is local
    assert a_global["adv_influence"] > 5 * (a_local["adv_influence"] + 1e-6)


def test_sensitivity_audit_honors_explicit_perturbation_support():
    """The certified GLW threat model excludes the trusted halo from perturbations, so the
    audit must honor an explicit support mask rather than always using the hard-mask
    background."""
    source, mask = _sample()
    global_fn = make_global_context_editor(pull=0.6)

    hard_bg = audit_editor_sensitivity(global_fn, source, mask, eps=8.0, n_random=4, spsa_steps=4)
    no_support = audit_editor_sensitivity(
        global_fn, source, mask, eps=8.0, n_random=4, spsa_steps=4,
        perturb_mask=np.zeros_like(mask, dtype=bool),
    )

    assert hard_bg["adv_influence"] > 1.0
    assert no_support["adv_influence"] == 0.0


def test_wrapper_confers_deterministic_locality_on_a_global_editor():
    """Wrapping a sensitivity-VIOLATING editor makes beyond-window pixels have provably zero
    influence on the in-region edit -- a deterministic certificate for an arbitrary editor."""
    from scipy.ndimage import binary_dilation

    source, mask = _sample()
    base = make_global_context_editor(0.6)            # in-region edit depends on background
    margin = 4
    wrapped = make_certified_local_wrapper(base, margin=margin)

    beyond_W = ~binary_dilation(mask > 0.5, iterations=margin)
    e0 = wrapped(source, mask, 0.5)

    rng = np.random.default_rng(1)
    pert = source.copy()
    pert[beyond_W] = rng.integers(0, 256, (int(beyond_W.sum()), 3), dtype=np.uint8)  # arbitrary
    e1 = wrapped(pert, mask, 0.5)

    # The IN-REGION edit is provably unchanged by beyond-window pixels (the certificate).
    # (Outside the mask, the output reflects the perturbed source by footprint preservation.)
    region = mask > 0.5
    np.testing.assert_array_equal(e0[region], e1[region])

    # Sanity: the un-wrapped editor's in-region edit IS influenced by the same change.
    assert not np.array_equal(base(source, mask, 0.5)[region], base(pert, mask, 0.5)[region])


def test_seed_noise_control_zero_for_deterministic_positive_for_stochastic():
    """The sampling-noise floor is 0 for a deterministic editor and >0 for a stochastic one,
    so a trigger can be judged real only when it exceeds this floor."""
    source, mask = _sample()
    det = make_cpu_editor_fn(EDITOR_CFG, "additive")          # deterministic
    sto = make_global_noise_editor(sigma=4.0)                 # stochastic (re-samples each call)

    a_det = audit_editor_sensitivity(det, source, mask, eps=8.0, n_random=4, spsa_steps=4,
                                     seed_noise_samples=4)
    a_sto = audit_editor_sensitivity(sto, source, mask, eps=8.0, n_random=4, spsa_steps=4,
                                     seed_noise_samples=4)
    assert a_det["seed_noise"] == 0.0
    assert a_sto["seed_noise"] > 0.0


def test_compare_editors_discriminates_certified_from_leaky():
    source, mask = _sample()
    editors = {
        "ours": make_cpu_editor_fn(EDITOR_CFG, "additive"),
        "relight": make_global_relight_editor(),
        "noise": make_global_noise_editor(sigma=3.0),
    }
    rep = compare_editors(source, mask, 0.5, editors, threshold=0.5)
    assert rep["ours"]["support_frac"] < 0.2          # leakage confined to feather band
    assert rep["relight"]["support_frac"] > 0.9
    assert rep["noise"]["support_frac"] > rep["ours"]["support_frac"]
