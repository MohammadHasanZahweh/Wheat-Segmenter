"""
Baseline ML for Wheat Segmentation (Random Forest, SVM)

Overview
- Pixel-level binary segmentation (wheat vs. non-wheat) using scikit-learn.
- Builds features from Sentinel-2 tiles provided by WheatTilesDataset.
- Samples ~1% of tiles (stratified by region and wheat coverage) and a capped
  number of valid pixels per tile to keep training fast and memory-friendly.

Design
- Dataset: Reuses WheatTilesDataset (see wheat_segmenter.py). We request
  temporal layout so each item returns x with shape [T,B,H,W], valid_mask [1,H,W],
  and wheat_mask [1,H,W]. Per-pixel features are the flattened channels
  (T*B) at each spatial location (H*W).
- Sampling tiles: Uses StratifiedRandomSubset from stratified_sampler.py to
  approximate a 1% subset while keeping region proportions and wheat-coverage
  diversity.
- Sampling pixels: Within each chosen tile, we sample up to --pixels-per-tile
  valid pixels. Optionally class-balance positives/negatives.
- Model: RandomForestClassifier or SVC (with probability=True). For SVM a
  StandardScaler is applied. RF does not require scaling.
- Split: Train/val split at tile level (to avoid pixel leakage across splits).

Metrics
- F1-score (binary) and IoU (intersection-over-union) computed on the pixel
  predictions of the held-out validation tiles.

Usage (PowerShell examples)
  # 1) Install dependencies (ensure scikit-learn is available)
  #    Activate venv first as per README2.md
  #    pip install -r requirements.txt

  # 2) Train and evaluate a Random Forest on ~1% of tiles
  python baseline_ml.py \
      --root "C:\\Users\\Administrator\\Desktop\\preprocessed_data" \
      --year 2020 \
      --model rf \
      --fraction 0.01 \
      --pixels-per-tile 4096 \
      --seed 42

  # 3) Train and evaluate an SVM (RBF kernel)
  python baseline_ml.py \
      --root "C:\\Users\\Administrator\\Desktop\\preprocessed_data" \
      --year 2020 \
      --model svm \
      --svm-kernel rbf \
      --svm-C 1.0 \
      --svm-gamma scale \
      --fraction 0.01 \
      --pixels-per-tile 4096 \
      --seed 42

Notes
- If labels are extremely imbalanced, consider --balance-pixels to draw a
  roughly equal number of wheat/non-wheat pixels per tile (subject to
  availability and the --pixels-per-tile cap).
- Increase --pixels-per-tile or --fraction if training is too fast and
  results are too noisy; decrease if you hit memory/time limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List

import argparse
import math
import numpy as np

from wheat_segmenter import WheatTilesDataset
from stratified_sampler import StratifiedRandomSubset


# ---------- Utility: metrics ----------
def f1_iou(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute F1 and IoU for binary labels (0/1), ignoring invalids upstream.

    y_true: shape [N], values in {0,1}
    y_pred: shape [N], values in {0,1}
    """
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    # F1 = 2TP / (2TP + FP + FN)
    denom_f1 = (2 * tp + fp + fn)
    f1 = (2.0 * tp / denom_f1) if denom_f1 > 0 else 0.0
    # IoU = TP / (TP + FP + FN)
    denom_iou = (tp + fp + fn)
    iou = (tp / denom_iou) if denom_iou > 0 else 0.0
    return f1, iou


# ---------- Feature extraction ----------
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
    - pixels_per_tile: cap total sampled valid pixels from this tile
    - balance: if True, attempt to sample equal positives/negatives
    - rng: numpy Generator

    Returns: (X, y) with shapes [M, T*B] and [M]
    """
    T, B, H, W = x_tb_hw.shape
    C = T * B

    valid_mask = (valid_hw > 0.5)
    wheat_mask = (wheat_hw > 0.5)

    if valid_mask.sum() == 0:
        return np.empty((0, C), dtype=np.float32), np.empty((0,), dtype=np.uint8)

    # Flatten spatial dimensions
    x_flat = x_tb_hw.reshape(T * B, H * W).T  # [H*W, C]
    valid_idx = np.flatnonzero(valid_mask.reshape(-1))
    labels = wheat_mask.reshape(-1).astype(np.uint8)

    # Consider only valid pixels
    valid_labels = labels[valid_idx]

    # If we want balance, sample positives and negatives separately
    if balance:
        pos_idx = valid_idx[valid_labels == 1]
        neg_idx = valid_idx[valid_labels == 0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            # Fall back to unbalanced sampling
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
    dataset: WheatTilesDataset,
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
        x = item["x"].numpy()  # [T,B,H,W]
        valid = item["valid_mask"].numpy()[0]  # [H,W]
        wheat = item["wheat_mask"].numpy()[0]  # [H,W]
        X_i, y_i = extract_pixels_from_item(x, valid, wheat, pixels_per_tile, balance, rng)
        if X_i.size > 0:
            xs.append(X_i)
            ys.append(y_i)
    if not xs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.uint8)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


# ---------- Train/eval pipeline ----------
@dataclass
class Config:
    root: str
    year: str
    regions: list[str] | None
    months: tuple[int, ...]
    fraction: float
    pixels_per_tile: int
    balance_pixels: bool
    seed: int
    model: str  # 'rf' or 'svm'
    # RF
    rf_estimators: int
    rf_max_depth: int | None
    # SVM
    svm_kernel: str
    svm_C: float
    svm_gamma: str | float
    # Misc
    val_ratio: float
    save_model: str | None


def load_dataset(cfg: Config) -> WheatTilesDataset:
    ds = WheatTilesDataset(
        root_preprocessed=cfg.root,
        year=cfg.year,
        regions=cfg.regions,
        month_order=cfg.months,
        temporal_layout=True,   # keep [T,B,H,W]
        normalize=True,
        band_stats=None,
        require_complete=True,
        target_bands=None,
        target_size=(64, 64),
        size_policy="pad",
        probe_limit=12,
    )
    return ds


def sample_tile_indices(ds: WheatTilesDataset, fraction: float, seed: int) -> List[int]:
    sampler = StratifiedRandomSubset(ds, fraction=fraction, n_bins=5, seed=seed)
    return list(iter(sampler))


def split_train_val(indices: Sequence[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.array(indices, dtype=np.int64)
    rng.shuffle(idx)
    n = len(idx)
    n_val = max(1, int(round(val_ratio * n))) if n > 1 else 0
    val_idx = idx[:n_val].tolist() if n_val > 0 else []
    train_idx = idx[n_val:].tolist()
    if not train_idx and val_idx:
        # Ensure we have at least some training if tiny sample
        train_idx, val_idx = val_idx, []
    return train_idx, val_idx


def train_and_eval(cfg: Config) -> None:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import joblib

    print("Loading dataset...")
    ds = load_dataset(cfg)
    print(f"Indexed tiles: {len(ds)}")

    print(f"Sampling ~{cfg.fraction*100:.2f}% of tiles (stratified)...")
    tile_indices = sample_tile_indices(ds, cfg.fraction, cfg.seed)
    print(f"Sampled tiles: {len(tile_indices)}")
    if len(tile_indices) == 0:
        raise RuntimeError("Sampler returned 0 tiles. Consider increasing --fraction or check data.")

    train_tiles, val_tiles = split_train_val(tile_indices, cfg.val_ratio, cfg.seed)
    print(f"Train tiles: {len(train_tiles)}, Val tiles: {len(val_tiles)}")

    print("Building training pixel matrix...")
    X_train, y_train = build_xy_from_tiles(
        ds, train_tiles, cfg.pixels_per_tile, cfg.balance_pixels, cfg.seed
    )
    print(f"Train pixels: {len(y_train)} | features: {X_train.shape[1] if X_train.size else 0}")
    if len(y_train) == 0:
        raise RuntimeError("No training pixels extracted. Increase --pixels-per-tile or adjust data.")

    if cfg.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=cfg.rf_estimators,
            max_depth=cfg.rf_max_depth,
            n_jobs=-1,
            random_state=cfg.seed,
            class_weight=None,
        )
    elif cfg.model == "svm":
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel=cfg.svm_kernel, C=cfg.svm_C, gamma=cfg.svm_gamma, probability=True, random_state=cfg.seed),
        )
    else:
        raise ValueError("Unsupported model. Use 'rf' or 'svm'.")

    print(f"Training model: {cfg.model}")
    clf.fit(X_train, y_train)

    # Evaluate on validation tiles (if any)
    if val_tiles:
        print("Building validation pixel matrix...")
        X_val, y_val = build_xy_from_tiles(
            ds, val_tiles, cfg.pixels_per_tile, False, cfg.seed + 1
        )
        print(f"Val pixels: {len(y_val)}")
        if len(y_val) > 0:
            y_pred = clf.predict(X_val)
            f1, iou = f1_iou(y_val, y_pred)
            pos_rate = float(y_val.mean()) if len(y_val) > 0 else 0.0
            print(f"Validation: F1={f1:.4f} | IoU={iou:.4f} | PosRate={pos_rate:.3f}")
        else:
            print("Validation set had 0 pixels after filtering.")
    else:
        print("No validation tiles (too small sample); reporting train metrics only.")
        y_pred_tr = clf.predict(X_train)
        f1, iou = f1_iou(y_train, y_pred_tr)
        pos_rate = float(y_train.mean())
        print(f"Train: F1={f1:.4f} | IoU={iou:.4f} | PosRate={pos_rate:.3f}")

    if cfg.save_model:
        outp = Path(cfg.save_model)
        outp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, outp)
        print(f"Saved model to {str(outp)}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Baseline ML (RF/SVM) for wheat segmentation")
    p.add_argument("--root", required=True, help="Preprocessed root containing data/ and label/")
    p.add_argument("--year", required=True, help="Year subfolder under data/ and label/")
    p.add_argument("--regions", nargs="*", default=None, help="Region ids (strings). If omitted, use all.")
    p.add_argument(
        "--months", nargs="*", type=int, default=[11,12,1,2,3,4,5,6,7],
        help="Months to include, e.g. 11 12 1 2 ..."
    )
    p.add_argument("--fraction", type=float, default=0.01, help="Fraction of tiles to sample (~1%)")
    p.add_argument("--pixels-per-tile", type=int, default=4096, help="Max valid pixels sampled per tile")
    p.add_argument("--balance-pixels", action="store_true", help="Class-balance pixel sampling within tiles")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio at tile level")
    p.add_argument("--model", choices=["rf", "svm"], default="rf", help="Choose baseline model")
    # RF
    p.add_argument("--rf-estimators", type=int, default=200, help="RandomForest n_estimators")
    p.add_argument("--rf-max-depth", type=int, default=None, help="RandomForest max_depth (None=unlimited)")
    # SVM
    p.add_argument("--svm-kernel", default="rbf", help="SVM kernel (rbf, linear, poly, sigmoid)")
    p.add_argument("--svm-C", type=float, default=1.0, help="SVM C parameter")
    p.add_argument("--svm-gamma", default="scale", help="SVM gamma (scale, auto, or float)")
    # Save
    p.add_argument("--save-model", default=None, help="Optional path to save the trained model (.joblib)")

    args = p.parse_args()

    # Convert args to Config
    months = tuple(int(m) for m in args.months)
    rf_max_depth = None if args.rf_max_depth in (None, 0) else int(args.rf_max_depth)

    # Parse svm_gamma: float or string
    try:
        svm_gamma: str | float = float(args.svm_gamma)
    except ValueError:
        svm_gamma = str(args.svm_gamma)

    return Config(
        root=str(args.root),
        year=str(args.year),
        regions=[str(r) for r in args.regions] if args.regions else None,
        months=months,
        fraction=float(args.fraction),
        pixels_per_tile=int(args.pixels_per_tile),
        balance_pixels=bool(args.balance_pixels),
        seed=int(args.seed),
        model=str(args.model),
        rf_estimators=int(args.rf_estimators),
        rf_max_depth=rf_max_depth,
        svm_kernel=str(args.svm_kernel),
        svm_C=float(args.svm_C),
        svm_gamma=svm_gamma,
        val_ratio=float(args.val_ratio),
        save_model=str(args.save_model) if args.save_model else None,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_and_eval(cfg)
