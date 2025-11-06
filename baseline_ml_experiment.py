"""
Enhanced Baseline ML for Wheat Segmentation (Random Forest, SVM)

- Train on a small stratified fraction of ALL tiles (default 1%).
- Test on a separate stratified fraction of the REMAINING tiles (default 25%).

Usage (PowerShell examples)
  # Random Forest, train 1%, test 25% of remaining
  python baseline_ml_experiment.py \
      --root "C:\\Users\\Administrator\\Desktop\\preprocessed_data" \
      --year 2020 \
      --model rf \
      --train-fraction 0.01 \
      --test-fraction 0.25 \
      --pixels-per-tile 4096 \
      --seed 42

  # SVM (RBF)
  python baseline_ml_experiment.py \
      --root "C:\\Users\\Administrator\\Desktop\\preprocessed_data" \
      --year 2020 \
      --model svm \
      --svm-kernel rbf \
      --svm-C 1.0 \
      --svm-gamma scale \
      --train-fraction 0.01 \
      --test-fraction 0.25 \
      --pixels-per-tile 4096 \
      --seed 42
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, List

import argparse
import numpy as np
from torch.utils.data import Subset

from wheat_segmenter import WheatTilesDataset
from stratified_sampler import StratifiedRandomSubset


# ---------- Utility: metrics ----------
def f1_iou(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
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


# ---------- Feature extraction ----------
def extract_pixels_from_item(
    x_tb_hw: np.ndarray,
    valid_hw: np.ndarray,
    wheat_hw: np.ndarray,
    pixels_per_tile: int,
    balance: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
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
    dataset: WheatTilesDataset,
    tile_indices: Sequence[int],
    pixels_per_tile: int,
    balance: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
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


# ---------- Train/eval pipeline ----------
@dataclass
class Config:
    root: str
    year: str
    regions: list[str] | None
    months: tuple[int, ...]
    train_fraction: float
    test_fraction: float
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
    save_model: str | None


def load_dataset(cfg: Config) -> WheatTilesDataset:
    ds = WheatTilesDataset(
        root_preprocessed=cfg.root,
        year=cfg.year,
        regions=cfg.regions,
        month_order=cfg.months,
        temporal_layout=True,
        normalize=True,
        band_stats=None,
        require_complete=True,
        target_bands=None,
        target_size=(64, 64),
        size_policy="pad",
        probe_limit=12,
    )
    return ds


def train_and_eval(cfg: Config) -> None:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import joblib

    print("Loading dataset...")
    ds = load_dataset(cfg)
    print(f"Indexed tiles: {len(ds)}")

    # Train sampling (~fraction of ALL tiles)
    print(f"Sampling TRAIN ~{cfg.train_fraction*100:.2f}% of tiles (stratified)...")
    train_tiles = list(iter(StratifiedRandomSubset(ds, fraction=cfg.train_fraction, n_bins=5, seed=cfg.seed)))
    print(f"Train tiles: {len(train_tiles)}")
    if len(train_tiles) == 0:
        raise RuntimeError("Train sampler returned 0 tiles. Increase --train-fraction or check data.")

    # Test sampling from remaining
    all_ids = set(range(len(ds)))
    remaining = sorted(all_ids.difference(set(train_tiles)))
    if len(remaining) == 0:
        raise RuntimeError("No remaining tiles to sample test set from. Lower --train-fraction.")
    rem_subset = Subset(ds, remaining)
    print(f"Sampling TEST ~{cfg.test_fraction*100:.2f}% of remaining tiles (stratified)...")
    test_sampler = StratifiedRandomSubset(rem_subset, fraction=cfg.test_fraction, n_bins=5, seed=cfg.seed + 7)
    val_tiles = [remaining[i] for i in iter(test_sampler)]
    print(f"Test tiles: {len(val_tiles)} (sampled from {len(remaining)} remaining)")

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

    # Evaluate on test tiles
    if val_tiles:
        print("Building test pixel matrix...")
        X_val, y_val = build_xy_from_tiles(
            ds, val_tiles, cfg.pixels_per_tile, False, cfg.seed + 1
        )
        print(f"Test pixels: {len(y_val)}")
        if len(y_val) > 0:
            y_pred = clf.predict(X_val)
            f1, iou = f1_iou(y_val, y_pred)
            pos_rate = float(y_val.mean()) if len(y_val) > 0 else 0.0
            print(f"Test: F1={f1:.4f} | IoU={iou:.4f} | PosRate={pos_rate:.3f}")
        else:
            print("Test set had 0 pixels after filtering.")
    else:
        print("No test tiles; reporting train metrics only.")
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
    p = argparse.ArgumentParser(description="Enhanced Baseline ML (RF/SVM) for wheat segmentation")
    p.add_argument("--root", required=True, help="Preprocessed root containing data/ and label/")
    p.add_argument("--year", required=True, help="Year subfolder under data/ and label/")
    p.add_argument("--regions", nargs="*", default=None, help="Region ids (strings). If omitted, use all.")
    p.add_argument(
        "--months", nargs="*", type=int, default=[11,12,1,2,3,4,5,6,7],
        help="Months to include, e.g. 11 12 1 2 ..."
    )

    # Fractions
    p.add_argument("--train-fraction", type=float, default=0.01, help="Fraction of ALL tiles for training")
    p.add_argument("--test-fraction", type=float, default=0.25, help="Fraction of REMAINING tiles for testing")

    p.add_argument("--pixels-per-tile", type=int, default=4096, help="Max valid pixels sampled per tile")
    p.add_argument("--balance-pixels", action="store_true", help="Class-balance pixel sampling within tiles")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
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
        train_fraction=float(args.train_fraction),
        test_fraction=float(args.test_fraction),
        pixels_per_tile=int(args.pixels_per_tile),
        balance_pixels=bool(args.balance_pixels),
        seed=int(args.seed),
        model=str(args.model),
        rf_estimators=int(args.rf_estimators),
        rf_max_depth=rf_max_depth,
        svm_kernel=str(args.svm_kernel),
        svm_C=float(args.svm_C),
        svm_gamma=svm_gamma,
        save_model=str(args.save_model) if args.save_model else None,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_and_eval(cfg)


