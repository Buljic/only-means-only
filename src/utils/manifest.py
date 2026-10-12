"""Artifact hash manifests for reproducibility (Phase 6).

A manifest maps every file under a root to its SHA-256 + size, so datasets, checkpoints, ONNX,
run JSONs and figures can be fingerprinted and later verified, so historical numbers stop living
only in Markdown. Torch-free and dependency-light (hashlib/json/pathlib).
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(root: str | Path, patterns: tuple[str, ...] = ("**/*",)) -> dict:
    """Fingerprint every FILE under ``root`` (recursively by default): ``{sha256, size_bytes}``
    keyed by POSIX relpath, sorted for determinism. Directories are skipped. Per-file digests
    run on a thread pool (the staged iHarmony4 tree is ~214k files; the serial pass was the
    slowest remaining staging step); the output dict is identical to the serial version."""
    root = Path(root)
    files: dict[str, dict] = {}
    paths: list[Path] = []
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file():
                paths.append(p)
    paths = sorted(set(paths), key=lambda p: p.relative_to(root).as_posix())
    rels = [p.relative_to(root).as_posix() for p in paths]
    with ThreadPoolExecutor(max_workers=min(24, (os.cpu_count() or 4))) as ex:
        digests = ex.map(sha256_file, paths)
        for rel, digest in zip(rels, digests):
            files[rel] = {"sha256": digest, "size_bytes": int((root / rel).stat().st_size)}
    files = dict(sorted(files.items()))
    return {"root": str(root), "n_files": len(files), "files": files}


def write_manifest(manifest: dict, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def verify_manifest(root: str | Path, manifest: dict) -> dict:
    """Compare ``root`` against a stored ``manifest``. Returns ``{'missing','changed','extra','ok'}``:
    recorded files absent on disk, present-but-different-hash, current files not recorded, and an
    overall ``ok`` (no missing and no changed; extras alone do not fail)."""
    root = Path(root)
    recorded = manifest.get("files", {})
    missing, changed = [], []
    for rel, info in recorded.items():
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif sha256_file(p) != info.get("sha256"):
            changed.append(rel)
    current = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    extra = sorted(current - set(recorded))
    return {"missing": sorted(missing), "changed": sorted(changed), "extra": extra,
            "ok": not missing and not changed}
