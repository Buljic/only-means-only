# Locality and sensitivity: two different guarantees

The manuscript separates two claims that are easy to conflate. This note fixes their
definitions, the assumptions each one rests on, and what the shipped CPU checks actually test.

## 1. Footprint (compositing locality)

Definition. An editor has footprint locality outside a set `S` if, for every pixel `p` in `S`,
the output equals the source byte for byte: `edit(p) == source(p)`.

How it is obtained. `src/editor/cpu_editor.py::composite_local` blends the edited image into
the source through a feathered mask with a hard threshold: pixels where
`feather(mask, r) < outside_threshold` get blend weight 0, so the result is exactly the
source there. `src/eval/locality.py::LocalityCertificate` makes the set explicit, and
`verify_zero_leakage` checks on a concrete pair that the maximum deviation over that set is 0.

Key distinction (R_target versus R_write). It is tempting to state equality outside the binary
mask `mask > 0.5`. That is not what the code guarantees, and the manuscript uses the correct
wording: feathering can give a pixel outside the binary mask a positive blend weight, in which
case the output there is not forced to equal the source. The certified set is the feathered
write support, `{p : feather(mask, r)(p) < outside_threshold}`, and the equality holds only on
that set. Any statement about pixels outside the binary mask must say which of the two regions
it means.

Assumptions. None beyond the arithmetic of the blend: the guarantee is structural and does not
depend on any trained network, and it holds for any edited image and any strength.

## 2. Sensitivity (exterior input to interior output)

Definition. Sensitivity bounds how much a change to pixels outside a trusted window `W` can
change the pixels written inside the region. It is a statement about information flow, not
about which pixels are written.

How it is obtained. In the shipped editors the exterior reaches the write only through a gated
global channel. The gate `gamma` bounds that channel: `global_sensitivity_bound()` returns
`2 * gamma * sqrt(3)` in per-pixel L2 units, and `certified_influence()` returns
`min(2 * |gamma|, L_cross * eps * sqrt(n_outside / n_region))` in mean-absolute
per-pixel-channel units, the tighter of an absolute gate bound and a Lipschitz bound. In the
typed variant the exterior also reaches a fixed six-channel statistic (per-channel RGB mean and
standard deviation); the unrestricted learned low-pass branch is disabled.

Assumptions. These are required for the bound as stated:

- a fixed mask and a fixed trusted window `W = dilate(region, halo)`;
- finite weights;
- circular padding for the FFT-exact per-layer operator norm (`model_certified_lipschitz`);
  with zero padding the FFT value is a close estimate only;
- real arithmetic. The computed norm uses float32 FFT and SVD and is NOT an outward-rounded
  machine certificate. A quantized output needs an additive rounding allowance.

Limits. The certified set is a restricted-architecture guarantee. The paper reports the mean
bound as below a separately measured edit magnitude for only 2 of 5 checkpoints, so the average
usefulness is partial, not general. The per-input PGD check in `src/eval/certify.py` is a
consistency check and a lower-bound search, not a proof of soundness. None of this is a claim
about modern generative editors.

## What the shipped tests check

| Test file | What it verifies |
|---|---|
| `tests/test_locality.py` | zero leakage for the certified editor; a leaky editor FAILS the same check (negative control); footprint matches the feathered support; the sensitivity audit flags a global-context editor and clears a local one; the wrapper confers deterministic locality on an arbitrary editor. |
| `tests/test_glw.py` | gate 0 gives provably zero in-region sensitivity; the bound is linear in the gate; the tight bound is sound against random perturbations; the cross-region certificate is sound and tighter than the absolute bound. |
| `tests/test_certify.py` | the convolution bounds are valid upper bounds; circular FFT norm is sound and tighter than the sum bound; the Lipschitz editor keeps exact zero footprint; adversarial influence respects the certificate. |

## What is not shipped

The saved checkpoints, the paper's numeric tables, the retrain, the SOTA baseline adapters and
the certificate-tightness run over real images are not part of this snapshot. They rest on
local-only inputs and are described in the manuscript, not reproduced here.
