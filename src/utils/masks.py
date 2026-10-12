"""Trusted-window mask helper.

Single, correct definition of the trusted window ``W = dilate(region, halo)`` used
everywhere the certificate/threat model needs it (GLW/TypedGLW input prep, the audit's
perturbation support, CCI, the universal trigger, the locality wrapper, the certified
trainer).

Why this exists: the ``halo=0`` footgun
----------------------------------------
``scipy.ndimage.binary_dilation(mask, iterations=0)`` does **not** mean "no dilation". Per
SciPy semantics, ``iterations=0`` means *"dilate repeatedly until the result stops
changing"*, that is, until the whole array is filled. So calling the raw primitive with
``halo=0`` silently expands the trusted window to (essentially) the entire image. That is
catastrophic in two independent ways:

* **False quality win.** The local branch, which must only see the trusted window, is handed
  a near-global read. A ``halo=0`` pilot would look like a win purely because "local" secretly
  saw everything.
* **Unsound certificate.** The cross-region Lipschitz bound is defined against pixels *outside*
  ``W``. If ``W`` is the whole image there are no outside pixels, so the certificate is vacuous
  / unsound.

``dilate_window`` makes ``halo <= 0`` an **exact no-op** (``W == region``) and only calls
``binary_dilation`` for strictly positive halos. Route every trusted-window construction
through here; never call ``binary_dilation(..., iterations=halo)`` directly in ``src/``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def dilate_window(mask, halo, *, structure=None) -> np.ndarray:
    """Return the trusted window: ``mask`` dilated by exactly ``halo`` iterations.

    Args:
        mask: array-like. Coerced to boolean via ``> 0.5`` unless already boolean.
        halo: number of dilation iterations. ``halo <= 0`` returns the (boolean) mask
            unchanged, an exact no-op, never SciPy's "iterate to convergence".
        structure: optional structuring element forwarded to ``binary_dilation``.

    Returns:
        ``numpy.ndarray`` of ``bool``, same shape as ``mask``.
    """
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        m = m > 0.5
    h = int(halo)
    if h <= 0:
        return m.copy()
    return binary_dilation(m, structure=structure, iterations=h)
