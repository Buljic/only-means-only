"""Torch-free tests for the iHarmony4 split-leakage guard.

Kept separate from the dataset-importing tests (which pull in torch) so the staging + leakage
logic can be exercised without the torch stack. Uses
dummy byte files: staging only copies and counts by suffix -- it never decodes the images.
"""

import json

import pytest
from PIL import Image

from scripts.download_iharmony4 import _split_leakage, stage_iharmony4


def _touch(path, data: bytes = b"img"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_release(root, listing):
    """Build an official-style release tree. ``listing`` maps split -> list of
    (real_stem, mask_suffix, composite_suffix). Composite = ``{real}_{mask}_{composite}.jpg``,
    real = ``{real}.jpg``, mask = ``{real}_{mask}.png`` (progressive-suffix matchable)."""
    for split, samples in listing.items():
        rels = []
        for real, mask_s, comp_s in samples:
            comp_rel = f"HCOCO/composite_images/{real}_{mask_s}_{comp_s}.jpg"
            value = 30 + sum(real.encode("utf-8")) % 180
            for path, colour in (
                (root / comp_rel, value // 2),
                (root / f"HCOCO/real_images/{real}.jpg", value),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(colour, colour, colour)).save(path)
            mask_path = root / f"HCOCO/masks/{real}_{mask_s}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (8, 8), color=255).save(mask_path)
            rels.append(comp_rel)
        (root / f"IHD_{split}.txt").write_text("\n".join(rels) + "\n", encoding="utf-8")


# --- pure function ----------------------------------------------------------------------------

def test_split_leakage_pure_detects_shared():
    assert _split_leakage({"train": {"a", "b"}, "test": {"b", "c"}}) == {"b"}


def test_split_leakage_pure_disjoint_is_empty():
    assert _split_leakage({"train": {"a", "b"}, "test": {"c", "d"}}) == set()


# --- staging integration ----------------------------------------------------------------------

def test_clean_release_layout_stages_and_records_leakage_check(tmp_path):
    src = tmp_path / "release"
    _write_release(src, {
        "train": [("scene1", "1", "1"), ("scene2", "1", "1")],
        "test": [("scene3", "1", "1")],                       # distinct real -> no leakage
    })
    out = tmp_path / "out"
    counts = stage_iharmony4(src, out)

    # A validation split is carved from train by real-image identity (2 train identities
    # -> 1 to val, 1 stays in train). Every composite is still staged exactly once.
    assert counts == {"train": 1, "val": 1, "test": 1}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["layout"] == "iharmony4_release"
    assert manifest["split_leakage_check"] == "passed"
    assert manifest["val_holdout_fraction"] > 0
    assert manifest["splits"]["test"]["unique_real_images"] == 1
    assert (out / "test" / "target" / "HCOCO__scene3_1_1.jpg").is_file()


def test_val_split_is_carved_and_three_way_disjoint(tmp_path):
    """Val is carved from train by real-image identity; train / val / test share no real
    image, and every train composite lands in exactly one of {train, val}."""
    src = tmp_path / "release"
    train = [(f"scene{i}", "1", "1") for i in range(1, 9)]     # 8 distinct real images
    _write_release(src, {"train": train, "test": [("held", "1", "1")]})
    out = tmp_path / "out"
    counts = stage_iharmony4(src, out)

    assert counts["val"] > 0 and counts["train"] > 0
    assert counts["train"] + counts["val"] == 8                # all train composites accounted for
    assert counts["test"] == 1

    def _identities(split):
        return {p.name for p in (out / split / "target").iterdir()}

    tr, va, te = _identities("train"), _identities("val"), _identities("test")
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)   # three-way disjoint
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["splits"]) == {"train", "val", "test"}
    assert manifest["splits"]["val"]["list"] == "carved_from_train"


def test_leaked_release_layout_raises(tmp_path):
    src = tmp_path / "release"
    # scene1 appears in BOTH splits (train composite scene1_1_1, test composite scene1_2_1),
    # both resolving to real_images/scene1.jpg -> leakage.
    _write_release(src, {
        "train": [("scene1", "1", "1")],
        "test": [("scene1", "2", "1")],
    })
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="split leakage"):
        stage_iharmony4(src, out)


def test_leaked_layout_fails_before_copying(tmp_path):
    src = tmp_path / "release"
    _write_release(src, {
        "train": [("scene1", "1", "1")],
        "test": [("scene1", "2", "1")],
    })
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        stage_iharmony4(src, out)
    # fail-fast: no split trees copied before the leakage check fired.
    assert not (out / "train").exists()
    assert not (out / "test").exists()


def test_staging_writes_a_valid_hash_manifest(tmp_path):
    from src.utils.manifest import verify_manifest
    src = tmp_path / "release"
    _write_release(src, {
        "train": [("scene1", "1", "1"), ("scene3", "1", "1")],
        "test": [("scene2", "1", "1")],
    })
    out = tmp_path / "out"
    stage_iharmony4(src, out)

    hm_path = out / "hash_manifest.json"
    assert hm_path.is_file()
    hm = json.loads(hm_path.read_text(encoding="utf-8"))
    assert hm["n_files"] >= 1                     # fingerprints the staged images + manifest.json
    diff = verify_manifest(out, hm)
    assert diff["missing"] == [] and diff["changed"] == []
    assert diff["extra"] == ["hash_manifest.json"]   # only the manifest itself is not in itself
