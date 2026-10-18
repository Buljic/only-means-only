# Only Means Only: construction-unit audit of iHarmony4 and certified-locality checks

## The question

Image-harmonization benchmarks build several composites from one real target image. The
official iHarmony4 dataset description states that target images are kept out of the opposite
split. This repository contains the CPU tools to check the released **filename lists** at the
real-image construction unit rather than the composite-file unit, and to run two locality
checks on a controlled editor: composite footprint (which pixels are written) and input
sensitivity (how far exterior input can move the interior output). The two claims are
different and are defined separately in `docs/CERTIFICATE.md`.

The audit finding here is a list-level property. The **score** contrasts that a shared
background could induce are a separate question, and the manuscript that uses these tools
reports composition controls, an opposite-direction checkpoint, and an inconclusive
intervention. This repository does not claim a uniform causal penalty, a first discovery, or a
verified clean split.

## What is included

- `src/editor/` and `src/eval/locality.py`: the compositing and locality code, including
  `LocalityCertificate` and `verify_zero_leakage`.
- `src/eval/certify.py`, `src/models/lipschitz.py`, `src/models/glw_editor.py`: the
  input-sensitivity certificate (bound computation and an empirical consistency check).
- `scripts/dataset_identity_scan.py`: the construction-unit identity matcher (standard library
  only).
- `scripts/download_iharmony4.py`: local staging and a fail-closed split-leakage guard.
- `data/iharmony4_lists/`: the official iHarmony4 train and test filename lists and the derived
  corrected test list. License and hashes in `THIRD_PARTY_NOTICES.md`.
- `tests/`: regression tests for the matcher and for both certificates, with negative controls.

## What is not included

- No model weights, checkpoints, images, masks, or targets.
- No third-party baseline code or weights (DoveNet, Harmonizer, PCT-Net, DucoNet, DiffHarmony).
  Harmonizer is CC BY-NC-SA 4.0 at the pinned revision and is not redistributed here.
- The manuscript and supplement are included here as `paper.pdf` and `supplement.pdf`.
  Their LaTeX source is released with the preprint record, not in this repository.
- No local run logs or machine-specific state.
- The exposure retrain, the SOTA scoring, and the certificate-tightness run over real images;
  they rest on local-only inputs.

## Quickstart (CPU smoke, no data or network)

Requires Python 3.10 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
python -m pip install numpy==2.5.1 scipy==1.18.0 scikit-image==0.26.0 matplotlib==3.11.0 Pillow==12.3.0 pytest==9.1.1

# Matcher on a synthetic fixture (no dataset needed)
python scripts/dataset_identity_scan.py \
  --name smoke \
  --train tests/fixtures/identity_lists/SYNTHETIC_train_mini.txt \
  --test  tests/fixtures/identity_lists/SYNTHETIC_test_mini.txt \
  --scope-subdataset --json

# Certificate and footprint tests
python -m pytest -q
```

A GitHub Actions workflow at `.github/workflows/cpu-smoke.yml` runs the same path plus the
official-list count on every push and pull request.

## Reproduce the split counts

These commands reproduce 1,399 shared identities and 1,822 covered test composites from the
shipped lists. Full detail and expected values are in `docs/REPRODUCING_LISTS.md`.

```bash
python scripts/dataset_identity_scan.py \
  --name iharmony4 \
  --train data/iharmony4_lists/IHD_train.official.txt \
  --test  data/iharmony4_lists/IHD_test.official.txt \
  --scope-subdataset --json
```

The +1 decoded-target duplicate and the four residual perceptual pairs are NOT reproduced from
filenames alone; those require decoding images and are described in the manuscript.

## Dataset acquisition

The iHarmony4 images are not shipped. Get them from the authors and stage them locally:

- Dataset repository and lists: https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4
- Stage and validate a local copy:
  `python scripts/download_iharmony4.py --source <official-root> --out data/iHarmony4 --overwrite`

The staging script checks triplet sizes, refuses an empty foreground mask, and fails closed if
the same decoded target appears in two splits. For the full baseline comparison, obtain
Harmonizer and the other baselines from their own authors under their own terms; see
`THIRD_PARTY_NOTICES.md`.

## Certificate assumptions

`docs/CERTIFICATE.md` gives the definitions. In short:

- Footprint (compositing) is data independent: outside the feathered write support the output
  equals the source byte for byte. This holds only outside the feathered support, which can
  differ from the binary mask.
- Sensitivity is a restricted-architecture bound on exterior-input to interior-output influence.
  It assumes a fixed mask and window, finite weights, circular padding for the FFT-exact norm,
  and real arithmetic. The float32 FFT/SVD computation is not an outward-rounded machine
  certificate.
- The paper's mean sensitivity bound falls below a separately measured edit magnitude for only
  2 of 5 checkpoints, so the average usefulness is partial.

## Results limits

This repository supports the split audit and the two certificate checks. It does not establish a
causal leakage penalty, a general deduplication method, a new harmonization-quality method, or a
clean split. The absolute scores, the composition controls, the retrain, and the baseline
calibration gap live in the manuscript and are not reproduced here.

## Citation

`CITATION.cff` describes this repository as software. Authors:

| Author | Affiliation | ORCID |
| --- | --- | --- |
| Imad Buljić | Polytechnic Faculty, University of Zenica | 0009-0009-2723-2315 |
| Delaida Muminović | Polytechnic Faculty, University of Zenica | 0009-0002-6388-650X |

Cite this repository for the code and the lists. Cite the preprint record for the scientific
claims: [10.5281/zenodo.23404529](https://doi.org/10.5281/zenodo.23404529) (concept DOI, always the latest version; v2, Zenodo,
CC BY 4.0). The manuscript is also
included here as `paper.pdf` and `supplement.pdf`.

## Layout

```
src/editor/          compositing and the CPU editor
src/eval/locality.py footprint certificate and the sensitivity audit
src/eval/certify.py  input-sensitivity certificate
src/models/          Lipschitz nets and the GLW editor
src/utils/           image, mask and manifest helpers
scripts/             identity matcher, staging and leakage guard
tests/               regression tests and synthetic fixtures
data/iharmony4_lists official train/test lists and the corrected test list
docs/                certificate notes, the list reproduction guide, and the
                     retrain run records behind the departure disclosure
```

## License

The project's own code and lists are released under the MIT license (`LICENSE`).
Third-party material is referenced, not redistributed; see `THIRD_PARTY_NOTICES.md`.
