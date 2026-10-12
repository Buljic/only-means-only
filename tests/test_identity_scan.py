"""Regression tests for the construction-unit identity scan.

The fixtures under ``tests/fixtures/identity_lists/`` are SYNTHETIC. They exercise the name
parser and the train/test intersection count, including a negative control, and carry no
scientific claim. The official iHarmony4 lists are shipped separately under
``data/iharmony4_lists/`` and are used by the documented reproduction command, not here.
"""

from pathlib import Path

from scripts.dataset_identity_scan import identity, scan

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "identity_lists"


def _read(name: str) -> list[str]:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def test_scan_counts_shared_identity_and_covered_composite():
    res = scan(
        _read("SYNTHETIC_train_mini.txt"),
        _read("SYNTHETIC_test_mini.txt"),
        scope=True,
    )
    assert res["n_intersection_identities"] == 1
    assert res["n_test_composites_covered"] == 1
    assert res["intersection_sample"] == ["hcoco/c100"]


def test_scan_negative_control_disjoint_lists_have_no_intersection():
    res = scan(
        _read("SYNTHETIC_train_mini.txt"),
        _read("SYNTHETIC_test_disjoint.txt"),
        scope=True,
    )
    assert res["n_intersection_identities"] == 0
    assert res["n_test_composites_covered"] == 0


def test_identity_rule_joins_the_first_two_tokens_for_long_stems():
    assert identity("HCOCO/composite_images/c100_200_1.jpg", True) == "hcoco/c100"
    assert (
        identity("HAdobe5k/composite_images/Canon1DsMkIII_0010_a_1.jpg", True)
        == "hadobe5k/canon1dsmkiii_0010"
    )
