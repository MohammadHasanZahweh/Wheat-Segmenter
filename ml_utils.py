"""
Shared ML utilities for pixel-level wheat segmentation.

Functions:
- f1_iou: binary F1 and IoU metrics.
- extract_pixels_from_item: build per-pixel features from one dataset item.
- build_xy_from_tiles: aggregate pixel features over a set of tiles.

These mirror the helpers used in the baseline scripts so other trainers can
reuse the exact same feature/label extraction.
"""

from __future__ import annotations

from typing import Sequence, Tuple, List, Any, TYPE_CHECKING

import numpy as np

# Import for type hints only; avoid hard dependency at import time
if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
    from wheat_segmenter import WheatTilesDataset  # type: ignore
else:
    WheatTilesDataset = Any  # type: ignore


def f1_iou(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute F1 and IoU for binary labels (0/1).

    Args
    - y_true: [N] array in {0,1}
    - y_pred: [N] array in {0,1}
    Returns (f1, iou)
    """
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom_f1 = (2 * tp + fp + fn)
    f1 = (2.0 * tp / denom_f1) if denom_f1 > 0 else 0.0
    denom_iou = (tp + fp + fn)
    iou = (tp / denom_iou) if denom_iou > 0 else 0.0
    return f1, iou


def extract_pixels_from_item(
    x_tb_hw: np.ndarray,
    valid_hw: np.ndarray,
    wheat_hw: np.ndarray,
    pixels_per_tile: int,
    balance: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-pixel features and labels from one dataset item.

    - x_tb_hw: float array [T,B,H,W]
    - valid_hw: float/bool array [H,W] (1=valid)
    - wheat_hw: float/bool array [H,W] (1=wheat)
    Returns (X, y) -> ([M, T*B], [M])
    """
    T, B, H, W = x_tb_hw.shape
    C = T * B

    valid_mask = (valid_hw > 0.5)
    wheat_mask = (wheat_hw > 0.5)

    if valid_mask.sum() == 0:
        return np.empty((0, C), dtype=np.float32), np.empty((0,), dtype=np.uint8)

    x_flat = x_tb_hw.reshape(T * B, H * W).T  # [H*W, C]
    valid_idx = np.flatnonzero(valid_mask.reshape(-1))
    labels = wheat_mask.reshape(-1).astype(np.uint8)

    valid_labels = labels[valid_idx]

    if balance:
        pos_idx = valid_idx[valid_labels == 1]
        neg_idx = valid_idx[valid_labels == 0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            take = min(pixels_per_tile, len(valid_idx))
            pick = rng.choice(valid_idx, size=take, replace=False) if take < len(valid_idx) else valid_idx
        else:
            half = pixels_per_tile // 2
            n_pos = min(half, len(pos_idx))
            n_neg = min(pixels_per_tile - n_pos, len(neg_idx))
            pos_pick = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos < len(pos_idx) else pos_idx
            neg_pick = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg < len(neg_idx) else neg_idx
            pick = np.concatenate([pos_pick, neg_pick])
            rng.shuffle(pick)
    else:
        take = min(pixels_per_tile, len(valid_idx))
        pick = rng.choice(valid_idx, size=take, replace=False) if take < len(valid_idx) else valid_idx

    X = x_flat[pick].astype(np.float32)
    y = labels[pick].astype(np.uint8)
    return X, y


def build_xy_from_tiles(
    dataset: "WheatTilesDataset",
    tile_indices: Sequence[int],
    pixels_per_tile: int,
    balance: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate per-pixel features and labels over a set of tiles."""
    rng = np.random.default_rng(seed)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for i in tile_indices:
        item = dataset[i]
        x = item["x"].numpy()
        valid = item["valid_mask"].numpy()[0]
        wheat = item["wheat_mask"].numpy()[0]
        X_i, y_i = extract_pixels_from_item(x, valid, wheat, pixels_per_tile, balance, rng)
        if X_i.size > 0:
            xs.append(X_i)
            ys.append(y_i)
    if not xs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.uint8)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y
