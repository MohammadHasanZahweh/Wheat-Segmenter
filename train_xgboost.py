"""
Train XGBoost baseline for wheat pixel segmentation.

Design
- Samples a stratified subset of tiles for training and a separate subset from
  the remaining tiles for testing.
- Extracts per-pixel features by flattening months×bands and labels from the
  wheat mask; valid-mask filters out no-data.

Usage (PowerShell)
  python train_xgboost.py \
    --root "C:\\Users\\Administrator\\Desktop\\preprocessed_data" \
    --year 2020 \
    --train-fraction 0.01 \
    --test-fraction 0.25 \
    --pixels-per-tile 4096 \
    --seed 42 \
    --n-estimators 400 --max-depth 8 --learning-rate 0.05 \
    --save-model runs/xgb_2020.joblib

Notes
- Requires xgboost to be installed (see requirements.txt). If installation is
  problematic on your Python version, consider the alternative in
  train_histgb.py which uses scikit-learn only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Sequence, List

import argparse
import numpy as np
from torch.utils.data import Subset

from wheat_segmenter import WheatTilesDataset
from stratified_sampler import StratifiedRandomSubset
from ml_utils import build_xy_from_tiles, f1_iou


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
    # XGB params
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    save_model: str | None


def load_dataset(cfg: Config) -> WheatTilesDataset:
    return WheatTilesDataset(
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


def train_and_eval(cfg: Config) -> None:
    import joblib
    from xgboost import XGBClassifier

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
    X_train, y_train = build_xy_from_tiles(ds, train_tiles, cfg.pixels_per_tile, cfg.balance_pixels, cfg.seed)
    print(f"Train pixels: {len(y_train)} | features: {X_train.shape[1] if X_train.size else 0}")
    if len(y_train) == 0:
        raise RuntimeError("No training pixels extracted. Increase --pixels-per-tile or adjust data.")

    clf = XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        tree_method="hist",
        objective="binary:logistic",
        n_jobs=-1,
        random_state=cfg.seed,
        eval_metric="logloss",
    )

    print("Training XGBoost...")
    clf.fit(X_train, y_train)

    # Evaluate on test tiles
    if val_tiles:
        print("Building test pixel matrix...")
        X_val, y_val = build_xy_from_tiles(ds, val_tiles, cfg.pixels_per_tile, False, cfg.seed + 1)
        print(f"Test pixels: {len(y_val)}")
        if len(y_val) > 0:
            y_pred = (clf.predict_proba(X_val)[:, 1] >= 0.5).astype(np.uint8)
            f1, iou = f1_iou(y_val, y_pred)
            pos_rate = float(y_val.mean()) if len(y_val) > 0 else 0.0
            print(f"Test: F1={f1:.4f} | IoU={iou:.4f} | PosRate={pos_rate:.3f}")
        else:
            print("Test set had 0 pixels after filtering.")
    else:
        print("No test tiles; reporting train metrics only.")
        y_pred_tr = (clf.predict_proba(X_train)[:, 1] >= 0.5).astype(np.uint8)
        f1, iou = f1_iou(y_train, y_pred_tr)
        pos_rate = float(y_train.mean())
        print(f"Train: F1={f1:.4f} | IoU={iou:.4f} | PosRate={pos_rate:.3f}")

    if cfg.save_model:
        outp = Path(cfg.save_model)
        outp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, outp)
        print(f"Saved model to {str(outp)}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="XGBoost baseline for wheat segmentation")
    p.add_argument("--root", required=True, help="Preprocessed root containing data/ and label/")
    p.add_argument("--year", required=True, help="Year subfolder under data/ and label/")
    p.add_argument("--regions", nargs="*", default=None, help="Region ids (strings). If omitted, use all.")
    p.add_argument("--months", nargs="*", type=int, default=[11,12,1,2,3,4,5,6,7], help="Months order")
    # Fractions
    p.add_argument("--train-fraction", type=float, default=0.01, help="Fraction of ALL tiles for training")
    p.add_argument("--test-fraction", type=float, default=0.25, help="Fraction of REMAINING tiles for testing")
    # Pixels
    p.add_argument("--pixels-per-tile", type=int, default=4096, help="Max valid pixels sampled per tile")
    p.add_argument("--balance-pixels", action="store_true", help="Class-balance pixel sampling within tiles")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    # XGB
    p.add_argument("--n-estimators", type=int, default=400, help="XGBoost n_estimators")
    p.add_argument("--max-depth", type=int, default=8, help="XGBoost max_depth")
    p.add_argument("--learning-rate", type=float, default=0.05, help="XGBoost learning_rate")
    p.add_argument("--subsample", type=float, default=0.8, help="XGBoost subsample")
    p.add_argument("--colsample-bytree", type=float, default=0.8, help="XGBoost colsample_bytree")
    # Save
    p.add_argument("--save-model", default=None, help="Optional path to save the trained model (.joblib)")

    args = p.parse_args()

    return Config(
        root=str(args.root),
        year=str(args.year),
        regions=[str(r) for r in args.regions] if args.regions else None,
        months=tuple(int(m) for m in args.months),
        train_fraction=float(args.train_fraction),
        test_fraction=float(args.test_fraction),
        pixels_per_tile=int(args.pixels_per_tile),
        balance_pixels=bool(args.balance_pixels),
        seed=int(args.seed),
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        save_model=str(args.save_model) if args.save_model else None,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_and_eval(cfg)
