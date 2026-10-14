# Third-party notices

This file records the third-party material that is redistributed in this snapshot, the
upstream material that is referenced but NOT redistributed, and the licenses of the runtime
dependencies. License facts below were checked on 2026-10-14.

## Redistributed third-party material

### iHarmony4 filename lists

- Files: `data/iharmony4_lists/IHD_train.official.txt`, `data/iharmony4_lists/IHD_test.official.txt`.
- Source: BCMI, Image-Harmonization-Dataset-iHarmony4, https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4
- License: MIT. The repository license was queried read-only through the GitHub API on
  2026-10-14 and reports SPDX `MIT`.
- Scope of redistribution: only the plain-text filename lists. No images, masks, targets or
  weights from the dataset are redistributed. The images in iHarmony4 are drawn from several
  upstream sources; the MIT label on the repository is not treated here as a license for those
  photographs, which is one reason only filenames are shipped.
- SHA-256 of the shipped files:
  - `IHD_train.official.txt`: `d423036efa0a3ae0fb1bdfa49482a1c147ecf82e1caa614860bf1bfdf993f9c9`
  - `IHD_test.official.txt`: `513ebcf6ec3cec01770c09483f75408386a12eb1bf10ecf805c782b3b289280a`
- Derived file: `data/iharmony4_lists/IHD_test.corrected.txt` is a first-party subset of
  `IHD_test.official.txt` (7,404 minus 1,823 composite paths; see `docs/REPRODUCING_LISTS.md`).
  It is distributed under the same MIT terms as its upstream input, with this notice preserved.

## Referenced but NOT redistributed

These are needed only for the full upstream-checkpoint scoring and comparison described in the
associated manuscript. This snapshot contains none of their code or weights.

- Harmonizer (ZHKKKe/Harmonizer). At the pinned revision
  `48ecd70becbff50ccaf576db0e64212dbc494e26` the README declares
  **CC BY-NC-SA 4.0** (verified read-only on 2026-10-14). Its code and weights are not
  redistributed here, and its license is not compatible with re-licensing any copy under the
  paper's CC BY 4.0. Use the authors' own distribution.
- DoveNet, PCT-Net, DucoNet, DiffHarmony: upstream baseline code and checkpoints. Not
  redistributed. The project license records identify PCT-Net as MPL-2.0 and the others as
  declaring no license at the inspected revision; each needs its own clearance before any
  redistribution. Obtain them from their authors.
- iHarmony4 images, masks and targets; ccHarmony test pairs. Not redistributed (ccHarmony is
  MIT-copyright BCMI but carries its own citation obligation). The supported path is user-side
  download plus the local staging script `scripts/download_iharmony4.py`.

## Runtime dependency licenses

| Package | Pin | License | Basis |
|---|---|---|---|
| numpy | 2.5.1 | BSD-3-Clause | upstream |
| scipy | 1.18.0 | BSD-3-Clause | installed metadata (license text present) |
| scikit-image | 0.26.0 | BSD-3-Clause | installed metadata (license text present) |
| matplotlib | 3.11.0 | PSF-style (Matplotlib license) | installed metadata (license text present) |
| Pillow | 12.3.0 | HPND (Pillow license) | upstream; installed metadata omits a License field |
| torch | 2.13.0 | BSD-3-Clause | upstream; installed metadata omits a License field |
| pytest | 9.1.1 | MIT | upstream; installed metadata omits a License field |

Rows marked "upstream" were not machine-verified from the installed metadata because those
distributions do not populate a License field. Verify them again if redistribution of the
dependency set itself is ever planned. The project's own code and lists are released under the
MIT license in `LICENSE`; the third-party material above keeps its own terms.
