"""Run-length decoding for Airbus Ship Detection (column-major / Fortran order)."""

from __future__ import annotations

import math
from typing import Union

import numpy as np


def rle_decode(mask_rle: Union[str, float, None], shape: tuple[int, int]) -> np.ndarray:
    """Decode Airbus-style RLE into a binary mask of shape (height, width).

    Positions are 1-based indices into the column-major flattened image, matching
    common Kaggle kernels for this competition.
    """
    h, w = shape
    flat = np.zeros(h * w, dtype=np.uint8)

    if mask_rle is None:
        return flat.reshape((h, w), order="F")

    if isinstance(mask_rle, float):
        if math.isnan(mask_rle):
            return flat.reshape((h, w), order="F")

    s = str(mask_rle).strip()
    if not s:
        return flat.reshape((h, w), order="F")

    parts = s.split()
    if len(parts) < 2 or len(parts) % 2 != 0:
        return flat.reshape((h, w), order="F")

    starts: list[int] = []
    lengths: list[int] = []
    for i in range(0, len(parts), 2):
        starts.append(int(parts[i]))
        lengths.append(int(parts[i + 1]))

    for start, length in zip(starts, lengths):
        # 1-based inclusive start in competition kernels -> 0-based in array
        lo = start - 1
        hi = lo + length
        lo = max(lo, 0)
        hi = min(hi, flat.size)
        if lo < hi:
            flat[lo:hi] = 1

    return flat.reshape((h, w), order="F")
