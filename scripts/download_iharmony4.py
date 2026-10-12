"""Stage iHarmony4 into the lightweight layout expected by ``IHarmony4Dataset``.

Accepted source layouts:

* already-normalized ``<split>/{composite,target,mask}``;
* the common iHarmony4 release layout with ``IHD_train.txt``/``IHD_test.txt`` and
  subdatasets such as ``HCOCO/composite_images``, ``HCOCO/real_images``, ``HCOCO/masks``.

The output is always normalized to ``<split>/{composite,target,mask}`` so local and GCP
runs use the same deterministic adapter path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Running `python scripts/download_iharmony4.py` puts scripts/ on sys.path, NOT the repo root, so
# the `from src.utils.manifest import ...` inside _finalize_manifest raised ModuleNotFoundError --
# AFTER the copy had already run, leaving a staged tree with no hash_manifest.json. The unit tests
# never caught it because pytest imports this module with the repo root already on sys.path; only
# the documented CLI invocation broke. Same one-liner the sibling scripts already use.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SPLITS = ("train", "test")
STAGED_SPLITS = ("train", "val", "test")
COMPOSITE_ALIASES = ("composite", "composites", "composite_images", "source", "input", "inputs")
TARGET_ALIASES = ("target", "targets", "real", "real_images", "harmonized", "gt", "ground_truth")
MASK_ALIASES = ("mask", "masks", "foreground_mask", "foreground_masks")
RELEASE_LISTS = {
    "train": ("IHD_train.txt", "train.txt", "train_list.txt", "training.txt"),
    "test": ("IHD_test.txt", "test.txt", "test_list.txt", "testing.txt"),
}


def _has_any(base: Path, names: tuple[str, ...]) -> bool:
    return any((base / n).is_dir() for n in names)


def _first_existing_dir(base: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = base / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"No directory under {base} matches any of: {names}")


def _iter_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _files_by_stem(directory: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in _iter_images(directory):
        if path.stem in out:
            raise ValueError(f"Ambiguous duplicate stem {path.stem!r} in {directory}")
        out[path.stem] = path
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _content_identity(root: Path) -> str:
    """Merkle-style identity of every staged sample byte, independent of source path."""
    # Sort by the RELATIVE POSIX STRING, never by Path objects. Path comparison is
    # case-insensitive on Windows and case-sensitive on POSIX, so `sorted(paths)` yields a
    # different order per OS -- and because this hash is order-dependent, staging on Windows and
    # validating on the Linux VM produced a false "content SHA-256 mismatch" (a hard failure in
    # paper mode). The real subdataset names trip exactly this: Windows orders Hday2night before
    # HFlickr, POSIX the reverse.
    files = sorted(
        (p for p in root.rglob("*")
         if p.is_file() and p.name not in {"manifest.json", "hash_manifest.json"}),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    items = [(p.relative_to(root).as_posix(), p) for p in files]
    h = hashlib.sha256()
    # Per-file digests in parallel (the official staged tree is ~214k files; the serial pass
    # took ~30-40 min and --validate-only re-runs it on every paper-mode launch). ex.map
    # preserves input order, so the final digest is byte-identical to the serial one.
    with ThreadPoolExecutor(max_workers=min(24, (os.cpu_count() or 4))) as ex:
        for rel, digest in ex.map(lambda item: (item[0], _sha256(item[1])), items):
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(digest.encode("ascii"))
            h.update(b"\n")
    return h.hexdigest()


def _finalize_manifest(root: Path, counts: dict[str, int]) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "omo.iharmony4_manifest.v2"
    manifest["validated_counts"] = counts
    manifest["content_sha256"] = _content_identity(root)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    from src.utils.manifest import build_manifest, write_manifest

    write_manifest(build_manifest(root), root / "hash_manifest.json")


def validate_staged_iharmony4(root: Path) -> dict[str, int]:
    """Validate triplets/splits and bind them to the staged byte-level manifest."""
    counts = _validate(root, require_val=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Staged iHarmony4 manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("content_sha256")
    if not expected:
        raise ValueError("Staged iHarmony4 manifest has no content_sha256")
    actual = _content_identity(root)
    if str(expected).lower() != actual.lower():
        raise ValueError(
            f"Staged iHarmony4 content SHA-256 mismatch: expected {expected}, got {actual}"
        )
    if manifest.get("validated_counts") != counts:
        raise ValueError(
            "Staged iHarmony4 manifest counts do not match current data: "
            f"{manifest.get('validated_counts')} != {counts}"
        )
    return counts


def _validate_triplet(composite: Path, target: Path, mask: Path) -> None:
    with Image.open(composite) as c_img, Image.open(target) as t_img, Image.open(mask) as m_img:
        if c_img.size != t_img.size or c_img.size != m_img.size:
            raise ValueError(
                f"Mismatched iHarmony4 triplet dimensions: {composite}={c_img.size}, "
                f"{target}={t_img.size}, {mask}={m_img.size}"
            )
        extrema = m_img.convert("L").getextrema()
        if extrema is None or extrema[1] <= 0:
            raise ValueError(f"Empty iHarmony4 foreground mask: {mask}")


def _relative_identity(source: Path, target: Path) -> str:
    """Machine-independent identity for an official-release target image.

    The source-relative POSIX path, case-folded. Falls back to the bare filename if the target
    somehow resolves outside ``source``. Never the absolute path: that would make the derived
    train/val carve depend on the staging directory (see the call site).
    """
    try:
        rel = target.resolve().relative_to(source.resolve()).as_posix()
    except ValueError:
        rel = target.name
    return rel.casefold()


def _target_identity(path: Path) -> str:
    """Decoded RGB identity, catching the same target even when it was re-encoded."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        h = hashlib.sha256()
        h.update(f"{rgb.width}x{rgb.height}".encode("ascii"))
        h.update(rgb.tobytes())
        return h.hexdigest()


def _validate_target_disjointness(targets: dict[str, list[Path]]) -> dict[str, int | bool]:
    """Fail if the same decoded ground-truth image appears in any two staged splits."""
    def decoded_set(paths: list[Path]) -> set[str]:
        with ThreadPoolExecutor(max_workers=min(24, (os.cpu_count() or 4))) as ex:
            return set(ex.map(_target_identity, paths))

    hashes = {split: decoded_set(paths) for split, paths in targets.items()}
    overlaps = {
        f"{left}/{right}": hashes[left] & hashes[right]
        for i, left in enumerate(hashes)
        for right in list(hashes)[i + 1:]
        if hashes[left] & hashes[right]
    }
    if overlaps:
        raise ValueError(
            "iHarmony4 target leakage across staged splits: "
            + ", ".join(f"{pair}={len(values)}" for pair, values in overlaps.items())
        )
    result = {f"target_hashes_{split}": len(values) for split, values in hashes.items()}
    result["target_splits_disjoint"] = True
    return result


def _validate(root: Path, *, require_val: bool = False) -> dict[str, int]:
    counts: dict[str, int] = {}
    targets_by_split: dict[str, list[Path]] = {}
    present_splits = [split for split in STAGED_SPLITS if (root / split).is_dir()]
    missing_required = [split for split in SPLITS if split not in present_splits]
    if missing_required:
        raise FileNotFoundError(f"Missing iHarmony4 split(s): {missing_required}")
    if require_val and "val" not in present_splits:
        raise FileNotFoundError(
            "Missing iHarmony4 validation split: frontier selection requires a staged val/ "
            "that is disjoint from train and test"
        )
    for split in present_splits:
        base = root / split
        if not base.is_dir():
            raise FileNotFoundError(f"Missing iHarmony4 split: {base}")
        checks = {
            "composite": COMPOSITE_ALIASES,
            "target": TARGET_ALIASES,
            "mask": MASK_ALIASES,
        }
        missing = [label for label, names in checks.items() if not _has_any(base, names)]
        if missing:
            raise FileNotFoundError(f"{base} missing iHarmony4 directories: {missing}")
        composite_dir = _first_existing_dir(base, COMPOSITE_ALIASES)
        target_dir = _first_existing_dir(base, TARGET_ALIASES)
        mask_dir = _first_existing_dir(base, MASK_ALIASES)
        composites = _files_by_stem(composite_dir)
        targets = _files_by_stem(target_dir)
        masks = _files_by_stem(mask_dir)
        if set(composites) != set(targets) or set(composites) != set(masks):
            missing_target = sorted(set(composites) - set(targets))[:5]
            missing_mask = sorted(set(composites) - set(masks))[:5]
            extra_target = sorted(set(targets) - set(composites))[:5]
            extra_mask = sorted(set(masks) - set(composites))[:5]
            raise ValueError(
                f"Unpaired iHarmony4 files in {base}: missing_target={missing_target}, "
                f"missing_mask={missing_mask}, extra_target={extra_target}, extra_mask={extra_mask}"
            )
        count = len(composites)
        if count == 0:
            raise FileNotFoundError(f"No iHarmony4 composite images found in {composite_dir}")
        stems = list(composites.keys())
        items = [(composites[s], targets[s], masks[s]) for s in stems]
        with ThreadPoolExecutor(max_workers=min(24, (os.cpu_count() or 4))) as ex:
            list(ex.map(lambda t: _validate_triplet(*t), items))
        targets_by_split[split] = list(targets.values())
        counts[split] = count
    _validate_target_disjointness(targets_by_split)
    return counts


def _find_list_file(root: Path, split: str) -> Path | None:
    for name in RELEASE_LISTS[split]:
        p = root / name
        if p.is_file():
            return p
    return None


def _read_list(path: Path) -> list[Path]:
    entries: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0].replace("\\", "/")
        entries.append(Path(token))
    if not entries:
        raise FileNotFoundError(f"iHarmony4 split list is empty: {path}")
    return entries


_MATCH_INDEX_CACHE: dict[Path, dict[str, list[Path]]] = {}


def _stem_index(directory: Path) -> dict[str, list[Path]]:
    """Cached stem -> sorted-path-list index for one directory.

    The fallback in ``_matching_file`` previously ran a full ``iterdir()`` per lookup, which on
    the official release is ~42k files in the HCOCO masks/real_images directories alone, at
    ~65k composites -> billions of stat calls and hours of staging. Same defect family as the
    dataset adapter's old ``_matching`` (fixed with the same cached-index pattern); the source
    directories are immutable during one staging invocation, so a module-level cache is safe.
    """
    index = _MATCH_INDEX_CACHE.get(directory)
    if index is None:
        index = {}
        for p in sorted(directory.iterdir()):
            if p.is_file():
                index.setdefault(p.stem, []).append(p)
        _MATCH_INDEX_CACHE[directory] = index
    return index


def _matching_file(directory: Path, stem: str, preferred_suffix: str | None = None) -> Path | None:
    if preferred_suffix:
        direct = directory / f"{stem}{preferred_suffix}"
        if direct.is_file():
            return direct
    matches = _stem_index(directory).get(stem)
    return matches[0] if matches else None


def _official_real_candidate(root: Path, composite_rel: Path) -> Path | None:
    # NOTE: match ONLY via the progressive-suffix loop below (longest stem prefix down to
    # shortest). An earlier naive-split guess here (splitting the WHOLE relative path, including
    # directory segments, on "_") could silently reconstruct the WRONG filename whenever a
    # directory name happened to contain an underscore -- and if that wrong guess coincidentally
    # existed on disk, it would return the WRONG real/target image with no error, silently
    # mispairing a training sample. The loop below is path-aware (splits only the filename stem)
    # and tries every valid prefix length, so it cannot mis-guess a shorter/longer id boundary.
    rel = composite_rel.as_posix()
    candidates: list[Path] = []
    stem_parts = Path(rel).stem.split("_")
    parent = Path(rel).parent.as_posix().replace("composite_images", "real_images")
    for keep in range(len(stem_parts), 0, -1):
        candidates.append(root / parent / ("_".join(stem_parts[:keep]) + ".jpg"))
        candidates.append(root / parent / ("_".join(stem_parts[:keep]) + ".png"))
    matches = list(dict.fromkeys(candidate for candidate in candidates if candidate.is_file()))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous iHarmony4 real-image match for {composite_rel}: {matches}")
    return matches[0] if matches else None


def _official_mask_candidate(root: Path, composite_rel: Path) -> Path | None:
    # See the matching NOTE in _official_real_candidate: match ONLY via the progressive-suffix
    # loop, never a naive whole-path split, to avoid a silent wrong-file mismatch.
    rel = composite_rel.as_posix()
    candidates: list[Path] = []
    stem_parts = Path(rel).stem.split("_")
    parent = Path(rel).parent.as_posix().replace("composite_images", "masks")
    for keep in range(len(stem_parts), 0, -1):
        candidates.append(root / parent / ("_".join(stem_parts[:keep]) + ".png"))
        candidates.append(root / parent / ("_".join(stem_parts[:keep]) + ".jpg"))
    matches = list(dict.fromkeys(candidate for candidate in candidates if candidate.is_file()))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous iHarmony4 mask match for {composite_rel}: {matches}")
    return matches[0] if matches else None


def _release_triplet(root: Path, composite_rel: Path) -> tuple[Path, Path, Path]:
    composite = root / composite_rel
    if not composite.is_file():
        raise FileNotFoundError(f"Listed iHarmony4 composite image not found: {composite}")
    subroot = composite.parent.parent
    target_dir = _first_existing_dir(subroot, TARGET_ALIASES)
    mask_dir = _first_existing_dir(subroot, MASK_ALIASES)
    target = _matching_file(target_dir, composite.stem, composite.suffix)
    mask = _matching_file(mask_dir, composite.stem, ".png")
    if target is None:
        target = _official_real_candidate(root, composite_rel)
    if mask is None:
        mask = _official_mask_candidate(root, composite_rel)
    if target is None:
        raise FileNotFoundError(f"No iHarmony4 real/target image matched {composite_rel}")
    if mask is None:
        raise FileNotFoundError(f"No iHarmony4 foreground mask matched {composite_rel}")
    return composite, target, mask


def _safe_stem(rel: Path) -> str:
    parts = [p for p in rel.with_suffix("").parts if p not in {"composite", "composites", "composite_images"}]
    return "__".join(part.replace(" ", "_") for part in parts)


def _copy_triplet(composite: Path, target: Path, mask: Path, out: Path, split: str, name_stem: str) -> None:
    split_root = out / split
    for label, src in (("composite", composite), ("target", target), ("mask", mask)):
        suffix = src.suffix.lower() if src.suffix.lower() in IMAGE_SUFFIXES else ".png"
        dst = split_root / label / f"{name_stem}{suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _split_leakage(identities: dict[str, set[str]]) -> set[str]:
    """Return identities present in more than one split."""
    leaked: set[str] = set()
    names = list(identities)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            leaked.update(identities[left] & identities[right])
    return leaked


def _carve_validation(records: list[dict], fraction: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Deterministically hold out complete real-image identities from official train."""
    identities = sorted({record["identity"] for record in records})
    if len(identities) < 2:
        return records, []
    n_val = min(len(identities) - 1, max(1, int(round(len(identities) * fraction))))
    # Hash ordering avoids privileging lexical filename prefixes while remaining deterministic.
    ordered = sorted(identities, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    val_ids = set(ordered[:n_val])
    return (
        [record for record in records if record["identity"] not in val_ids],
        [record for record in records if record["identity"] in val_ids],
    )


def _normalized_validation_identities(root: Path, fraction: float = 0.2) -> set[str]:
    """Choose complete decoded-target identities to move from normalized train into val."""
    train = root / "train"
    target_dir = _first_existing_dir(train, TARGET_ALIASES)
    identities = sorted({_target_identity(path) for path in _iter_images(target_dir)})
    if len(identities) < 2:
        raise ValueError(
            "Normalized iHarmony4 source has no val/ and fewer than two unique train targets; "
            "cannot create disjoint non-empty train/val splits"
        )
    n_val = min(len(identities) - 1, max(1, int(round(len(identities) * fraction))))
    ordered = sorted(identities, key=lambda value: hashlib.sha256(value.encode("ascii")).hexdigest())
    return set(ordered[:n_val])


def _carve_normalized_validation(root: Path, val_identities: set[str]) -> None:
    """Move all triplets sharing a chosen target identity, preserving train directory aliases."""
    train = root / "train"
    composite_dir = _first_existing_dir(train, COMPOSITE_ALIASES)
    target_dir = _first_existing_dir(train, TARGET_ALIASES)
    mask_dir = _first_existing_dir(train, MASK_ALIASES)
    composites = _files_by_stem(composite_dir)
    targets = _files_by_stem(target_dir)
    masks = _files_by_stem(mask_dir)
    for stem, target in targets.items():
        if _target_identity(target) not in val_identities:
            continue
        for src in (composites[stem], target, masks[stem]):
            relative_dir = src.parent.relative_to(train)
            destination = root / "val" / relative_dir / src.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(destination))


def _stage_release_layout(source: Path, out: Path, max_per_split: int | None = None) -> dict[str, int]:
    list_files = {split: _find_list_file(source, split) for split in SPLITS}
    if any(p is None for p in list_files.values()):
        missing = [split for split, path in list_files.items() if path is None]
        raise FileNotFoundError(
            "Source is not normalized and is missing iHarmony4 split list(s): "
            + ", ".join(missing)
            + ". Expected IHD_train.txt/IHD_test.txt or train.txt/test.txt."
        )

    records_by_split: dict[str, list[dict]] = {}
    for split in SPLITS:
        entries = _read_list(list_files[split])  # type: ignore[arg-type]
        if max_per_split is not None:
            entries = entries[: max(0, int(max_per_split))]
        records = []
        for rel in entries:
            composite, target, mask = _release_triplet(source, rel)
            _validate_triplet(composite, target, mask)
            records.append({
                "rel": rel,
                "composite": composite,
                "target": target,
                "mask": mask,
                # Identity must be a property of the DATA, not of where it happens to sit.
                # This previously used the absolute resolved path, so `_carve_validation`
                # (which orders identities by sha256(identity)) produced a DIFFERENT
                # train/val split for the same dataset staged from a different directory --
                # and the evaluation is selected on that val split. Staging under one
                # directory on one machine and another elsewhere silently selected on different
                # data, and the split would not be reproducible elsewhere. The source-relative POSIX
                # path is stable across machines and operating systems for this layout.
                "identity": _relative_identity(source, target),
            })
        records_by_split[split] = records

    # Resolve every official target before making any destination tree.  This catches the stronger
    # real-identity leak even when train/test composites have different bytes or mask suffixes.
    official_identities = {
        split: {record["identity"] for record in records}
        for split, records in records_by_split.items()
    }
    leaked = _split_leakage(official_identities)
    if leaked:
        raise ValueError(
            f"iHarmony4 split leakage: {len(leaked)} real image identity/identities occur in "
            "both official train and test"
        )

    train_records, val_records = _carve_validation(records_by_split["train"])
    staged = {"train": train_records, "val": val_records, "test": records_by_split["test"]}
    counts: dict[str, int] = {}
    manifest: dict[str, object] = {
        "source": str(source),
        "layout": "iharmony4_release",
        "split_leakage_check": "passed",
        "val_holdout_fraction": 0.2,
        "splits": {},
    }
    for split, records in staged.items():
        if not records and split == "val":
            continue
        seen: set[str] = set()
        for copied, record in enumerate(records):
            rel = record["rel"]
            stem = _safe_stem(rel)
            if stem in seen:
                stem = f"{stem}__{copied:06d}"
            seen.add(stem)
            _copy_triplet(
                record["composite"], record["target"], record["mask"], out, split, stem
            )
        counts[split] = len(records)
        manifest["splits"][split] = {
            "list": "carved_from_train" if split == "val" else str(list_files[split]),
            "count": len(records),
            "unique_real_images": len({record["identity"] for record in records}),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return counts


def stage_iharmony4(source: Path, out: Path, *, overwrite: bool = False, max_per_split: int | None = None) -> dict[str, int]:
    source = source.resolve()
    out = out.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"iHarmony4 source not found: {source}")
    if source == out or source in out.parents or out in source.parents:
        raise ValueError(
            f"iHarmony4 --source and --out must not overlap (source={source}, out={out})"
        )
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {out}. Pass --overwrite to replace it.")
        shutil.rmtree(out)

    try:
        counts = _validate(source, require_val=False)
    except FileNotFoundError:
        _stage_release_layout(source, out, max_per_split=max_per_split)
    else:
        val_identities = None
        if not (source / "val").is_dir():
            val_identities = _normalized_validation_identities(source)
        shutil.copytree(source, out)
        if val_identities is not None:
            _carve_normalized_validation(out, val_identities)
        counts = _validate(out, require_val=True)
        manifest = {
            "source": str(source),
            "layout": "normalized",
            "val_holdout": "existing" if val_identities is None else "carved_by_decoded_target_identity",
            "val_holdout_fraction": 0.2,
            "splits": counts,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        _finalize_manifest(out, counts)
        return counts
    counts = _validate(out, require_val=True)
    _finalize_manifest(out, counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage/validate iHarmony4 dataset")
    parser.add_argument("--source", type=Path, default=None, help="existing staged iHarmony4 root")
    parser.add_argument("--out", type=Path, default=Path("data/iHarmony4"), help="destination root")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="replace --out if it already exists")
    parser.add_argument("--max-per-split", type=int, default=None, help="optional staging cap for smoke tests")
    args = parser.parse_args()

    if args.validate_only:
        counts = validate_staged_iharmony4(args.out)
        print(f"iHarmony4 layout OK: {args.out} ({counts})")
        return
    if args.source is None:
        raise SystemExit(
            "Provide --source pointing to either a normalized iHarmony4 root with train/test composite,target,mask "
            "directories or an official-style root with IHD_train.txt/IHD_test.txt and subdataset directories."
        )
    counts = stage_iharmony4(args.source, args.out, overwrite=args.overwrite, max_per_split=args.max_per_split)
    print(f"Staged iHarmony4 -> {args.out} ({counts})")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
