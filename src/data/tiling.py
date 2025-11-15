from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np


def extract_patches(x: np.ndarray, patch_size: int, stride: int) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Yield patches from array x of shape (C,H,W).

    Yields:
        (row, col, patch) where row/col are top-left indices in pixels.
    """
    C, H, W = x.shape
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            yield r, c, x[:, r : r + patch_size, c : c + patch_size]

