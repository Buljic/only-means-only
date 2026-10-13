# Reproducing the split-identity counts

The headline split numbers are recomputed from the two official iHarmony4 filename lists that
ship in `data/iharmony4_lists/`. The command is deterministic, CPU only, standard library only,
and needs no images and no network.

## Command

```bash
python scripts/dataset_identity_scan.py \
  --name iharmony4 \
  --train data/iharmony4_lists/IHD_train.official.txt \
  --test  data/iharmony4_lists/IHD_test.official.txt \
  --scope-subdataset --json
```

## Expected result

| Field | Value |
|---|---|
| `n_train_composites` | 65742 |
| `n_test_composites` | 7404 |
| `n_train_identities` | 35288 |
| `n_test_identities` | 4317 |
| `n_intersection_identities` | 1399 |
| `n_test_composites_covered` | 1822 |
| `test_coverage_fraction` | 0.24608319827120476 |
| `test_composites_per_identity_max` | 14 |

## Identity rule

The rule is fixed in `scripts/dataset_identity_scan.py::identity`:

1. take the filename, strip the extension and any directory path;
2. split the stem on underscores;
3. if there are fewer than three underscore tokens, raise (unexpected name);
4. the identity is the first token, or the first two tokens joined with an underscore when the
   name has four or more underscore tokens (this handles real-image ids that themselves contain
   an underscore, for example `Canon1DsMkIII_0010_a_1` to `Canon1DsMkIII_0010`);
5. case-fold the result;
6. with `--scope-subdataset`, prefix the leading path component (the sub-dataset) so names only
   compare within one sub-dataset.

The script is a post-hoc construction scan, not a pre-registered audit. Two different files with
the same name stem are not told apart, and the same photograph stored under a different name is
not found.

## The corrected list and the excluded set

`data/iharmony4_lists/IHD_test.corrected.txt` is the official test list with every composite of a
shared identity removed, plus the one decoded-target duplicate described below.

```bash
wc -l data/iharmony4_lists/IHD_test.corrected.txt       # 5581
# excluded = official minus corrected
comm -23 <(sort data/iharmony4_lists/IHD_test.official.txt) \
         <(sort data/iharmony4_lists/IHD_test.corrected.txt) | wc -l   # 1823
```

`1823 = 1822` path-identity composites (the `n_test_composites_covered` above) plus `1`
decoded-target duplicate.

## What this does not reproduce

- The one decoded-target duplicate (the 1,823rd exclusion) is not found by name. It was found by
  hashing decoded target RGB and is not reproduced from the shipped filename lists. The code that
  computes decoded-target identity is `scripts/download_iharmony4.py::_target_identity`, which
  needs the images.
- After correction, four perceptually confirmed cross-split pairs remain (reported in the
  manuscript). Those were found with a separate perceptual audit and are not reproduced here.
- The official train list is shipped as `IHD_train.official.txt`, which is byte-identical to the
  release's `IHD_train.txt` at this checkout.
