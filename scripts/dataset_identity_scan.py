#!/usr/bin/env python3
"""Post-hoc construction scan: source-identity overlap between published train/test lists.

CPU only, standard library only, read-only. Not a pre-registered audit.

Identity rule (name based). Take the file name, strip the extension and any directory.
Split on "_". Drop the last two tokens (the variant letter and the composite index for
ccHarmony; the mask id and the composite index for iHarmony4). The remaining leading tokens
are the source stem. identity = the first token, or the first two tokens joined with "_"
when the name has four or more underscore tokens. Examples:

    8D5U5594_a_1.jpg                 3 tokens -> 8D5U5594
    Canon1DsMkIII_0010_a_1.jpg       4 tokens -> Canon1DsMkIII_0010
    HCOCO/composite_images/c100034_658772_1.jpg   3 tokens -> c100034
    Hday2night/composite_images/d10091-20120914-152206_1_1.jpg -> d10091-20120914-152206

For iHarmony4 the identity is also scoped by sub-dataset (the leading path component),
matching src/eval/e1_adapters.py::background_identity. Pass --scope-subdataset.

Usage:
    dataset_identity_scan.py --name NAME --train TRAIN.txt --test TEST.txt [--scope-subdataset]
        [--cross-train OTHER_TRAIN.txt --cross-name OTHER] [--json]

Name-level matching only: two different files with the same name stem are not told apart,
and the same photograph under a different name is not found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def stem_tokens(line: str) -> tuple[str, list[str]]:
    """Return (subdataset, tokens) for one list line."""
    p = Path(line.strip())
    sub = p.parts[0] if len(p.parts) > 1 else ""
    return sub, p.stem.split("_")


VARIANT_SUFFIX = re.compile(r"-\d+$")


def identity(line: str, scope_subdataset: bool = False, rule: str = "tokens") -> str:
    sub, tok = stem_tokens(line)
    if rule == "stem":
        # Whole-image lists (e.g. DESOBA): identity = file stem; "stem-variant" also drops a
        # trailing "-N" variant index (web-shadow0248-1 -> web-shadow0248).
        ident = "_".join(tok).casefold()
        ident = VARIANT_SUFFIX.sub("", ident)
        return f"{sub.casefold()}/{ident}" if scope_subdataset else ident
    if len(tok) < 3:
        raise ValueError(f"unexpected name (fewer than 3 underscore tokens): {line!r}")
    ident = "_".join(tok[:2]) if len(tok) >= 4 else tok[0]
    ident = ident.casefold()
    return f"{sub.casefold()}/{ident}" if scope_subdataset else ident


def read_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def scan(train: list[str], test: list[str], scope: bool, rule: str = "tokens") -> dict:
    tr = [identity(x, scope, rule) for x in train]
    te = [identity(x, scope, rule) for x in test]
    tr_set, te_set = set(tr), set(te)
    both = tr_set & te_set
    covered = sum(1 for x in te if x in both)
    return {
        "n_train_composites": len(train),
        "n_test_composites": len(test),
        "n_train_identities": len(tr_set),
        "n_test_identities": len(te_set),
        "n_intersection_identities": len(both),
        "n_test_composites_covered": covered,
        "test_coverage_fraction": covered / len(test) if test else 0.0,
        "test_composites_per_identity_max": max(Counter(te).values()) if te else 0,
        "intersection_sample": sorted(both)[:5],
    }


def cross(test: list[str], other_train: list[str], scope: bool) -> dict:
    ot = {identity(x, scope) for x in other_train}
    te = [identity(x, scope) for x in test]
    both = set(te) & ot
    return {
        "n_test_identities": len(set(te)),
        "n_other_train_identities": len(ot),
        "n_intersection_identities": len(both),
        "n_test_composites_covered": sum(1 for x in te if x in both),
        "intersection_sample": sorted(both)[:5],
    }


def cc_file_check(root: Path, train: list[str], test: list[str]) -> dict:
    """ccHarmony file-level check: referenced files exist; real-image files shared across splits.

    Real file = <realid>.jpg (stem minus last two tokens), mask = <stem minus last token>.png, as in
    src/data/ccharmony_dataset.py. Byte-identical real files under different names are found by sha256.
    """
    import hashlib

    def real(n: str) -> str:
        return "_".join(Path(n).stem.split("_")[:-2]) + ".jpg"

    def mask(n: str) -> str:
        return "_".join(Path(n).stem.split("_")[:-1]) + ".png"

    out: dict = {}
    for tag, names in (("train", train), ("test", test)):
        out[f"{tag}_missing_composite"] = sorted(n for n in names if not (root / "composite_images" / n).is_file())
        out[f"{tag}_missing_real"] = sorted({real(n) for n in names if not (root / "real_images" / real(n)).is_file()})
        out[f"{tag}_missing_mask"] = sorted({mask(n) for n in names if not (root / "masks" / mask(n)).is_file()})
        out[f"{tag}_real_files"] = len({real(n) for n in names})
        out[f"{tag}_mask_files"] = len({mask(n) for n in names})
    tr_real, te_real = {real(n) for n in train}, {real(n) for n in test}
    out["shared_real_files"] = sorted(tr_real & te_real)
    tr_mask, te_mask = {mask(n) for n in train}, {mask(n) for n in test}
    out["shared_mask_files"] = sorted(tr_mask & te_mask)

    def sha(f: str) -> str:
        return hashlib.sha256((root / "real_images" / f).read_bytes()).hexdigest()

    th = {}
    for f in tr_real:
        if (root / "real_images" / f).is_file():
            th.setdefault(sha(f), []).append(f)
    out["byte_identical_real_across_splits"] = sorted(
        (f, th[sha(f)]) for f in te_real if (root / "real_images" / f).is_file() and sha(f) in th)
    out["unreferenced_real_files"] = sorted(
        f.name for f in (root / "real_images").iterdir() if f.name not in tr_real | te_real)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--scope-subdataset", action="store_true")
    ap.add_argument("--cross-train", type=Path, help="another dataset's train list for a cross-check")
    ap.add_argument("--cross-name", default="other")
    ap.add_argument("--rule", choices=["tokens", "stem"], default="tokens",
                    help="tokens (default, ccHarmony/iHarmony4) or stem (whole-image lists, strips -N variant suffix)")
    ap.add_argument("--cc-file-check", type=Path, metavar="CCHARMONY_ROOT",
                    help="also run the ccHarmony file-level check against this archive root")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    train, test = read_list(a.train), read_list(a.test)
    out = {"name": a.name, "label": "post-hoc construction scan, not a pre-registered audit",
           "rule": ("first token, or first two joined when >= 4 underscore tokens" if a.rule == "tokens"
                    else "file stem with trailing -N variant index dropped"),
           "scope_subdataset": a.scope_subdataset, "train_list": str(a.train), "test_list": str(a.test),
           "scan": scan(train, test, a.scope_subdataset, a.rule)}
    if a.cross_train:
        # The cross-check always uses the bare identity (no sub-dataset scope), so names compare across datasets.
        out["cross"] = {"vs": a.cross_name, "train_list": str(a.cross_train),
                        **cross(test, read_list(a.cross_train), False)}
    if a.cc_file_check:
        out["cc_file_check"] = cc_file_check(a.cc_file_check, train, test)
    if a.json:
        json.dump(out, sys.stdout, indent=2)
        print()
    else:
        print(f"[{a.name}] post-hoc construction scan (not a pre-registered audit)")
        for k, v in out["scan"].items():
            print(f"  {k}: {v}")
        if "cross" in out:
            print(f"  cross-check: {a.name} test vs {a.cross_name} train")
            for k, v in out["cross"].items():
                if k not in ("vs", "train_list"):
                    print(f"    {k}: {v}")
        for k, v in out.get("cc_file_check", {}).items():
            print(f"  file_check.{k}: {v if not isinstance(v, list) or len(v) <= 5 else str(len(v)) + ' items, first 5 ' + str(v[:5])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
