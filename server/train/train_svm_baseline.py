from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

import argparse
import os
import sys

import numpy as np
from torch.utils.data import Subset

# Ensure project root is on sys.path for ml_utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server.dataset.PatchDataset import WheatTilesDataset, StratifiedRandomSubset
from ml_utils import build_xy_from_tiles, f1_iou


@dataclass
class Config:
    root: str
    year: str
    regions: List[str] | None
    months: tuple[int, ...]
    train_fraction: float
    test_fraction: float
    pixels_per_tile: int
    balance_pixels: bool
    seed: int
    svm_kernel: str
    svm_C: float
    svm_gamma: str | float
    save_model: str | None
    use_meta_stats: bool


def load_dataset(cfg: Config) -> WheatTilesDataset:
    return WheatTilesDataset(
        root_preprocessed=cfg.root,
        year=cfg.year,
        regions=cfg.regions,
        month_order=cfg.months,
        temporal_layout=True,
        normalize=True,
        band_stats='auto' if cfg.use_meta_stats else None,
        meta_dir='./meta' if cfg.use_meta_stats else None,
        require_complete=True,
        target_bands=None,
        target_size=(64, 64),
        size_policy="pad",
        probe_limit=12,
    )


def train_and_eval(cfg: Config) -> dict[str, Any]:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    import joblib

    print("Loading dataset...")
    ds = load_dataset(cfg)
    print(f"Indexed tiles: {len(ds)}")

    print(f"Sampling TRAIN ~{cfg.train_fraction*100:.2f}% of tiles (stratified)...")
    train_tiles = list(iter(StratifiedRandomSubset(ds, fraction=cfg.train_fraction, n_bins=5, seed=cfg.seed)))
    print(f"Train tiles: {len(train_tiles)}")
    if len(train_tiles) == 0:
        raise RuntimeError("Train sampler returned 0 tiles. Increase --train-fraction or check data.")

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

    clf = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=cfg.svm_kernel,
            C=cfg.svm_C,
            gamma=cfg.svm_gamma,
            probability=True,
            random_state=cfg.seed,
        ),
    )

    print("Training SVM...")
    clf.fit(X_train, y_train)

    eval_split = "train"
    eval_pixels = len(y_train)
    test_pixels = 0
    y_eval = y_train
    if val_tiles:
        print("Building test pixel matrix...")
        X_val, y_val = build_xy_from_tiles(ds, val_tiles, cfg.pixels_per_tile, False, cfg.seed + 1)
        print(f"Test pixels: {len(y_val)}")
        test_pixels = len(y_val)
        if len(y_val) > 0:
            y_eval = y_val
            y_pred = clf.predict(X_val)
            eval_split = "test"
            eval_pixels = len(y_val)
        else:
            print("Test set had 0 pixels after filtering. Falling back to train metrics.")
            y_pred = clf.predict(X_train)
    else:
        print("No test tiles; reporting train metrics only.")
        y_pred = clf.predict(X_train)

    if cfg.save_model:
        outp = Path(cfg.save_model)
        outp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, outp)
        print(f"Saved model to {str(outp)}")
        model_path = str(outp)
    else:
        model_path = None

    f1, iou = f1_iou(y_eval, y_pred)
    pos_rate = float(y_eval.mean()) if len(y_eval) > 0 else 0.0
    if eval_split == "test":
        print(f"Test: F1={f1:.4f} | IoU={iou:.4f} | PosRate={pos_rate:.3f}")
    else:
        print(f"Train: F1={f1:.4f} | IoU={iou:.4f} | PosRate={pos_rate:.3f}")

    result: dict[str, Any] = {
        "status": "completed",
        "train_tiles": len(train_tiles),
        "test_tiles": len(val_tiles),
        "train_pixels": int(len(y_train)),
        "test_pixels": int(test_pixels),
        "evaluation_split": eval_split,
        "evaluation_pixels": int(eval_pixels),
        "metrics": {
            "f1": float(f1),
            "iou": float(iou),
            "positive_rate": float(pos_rate),
        },
    }
    if model_path:
        result["model_path"] = model_path
    return result


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="SVM baseline for wheat segmentation (train/test split)"
    )
    p.add_argument("--root", required=True, help="Preprocessed root containing data/ and label/")
    p.add_argument("--year", required=True, help="Year subfolder under data/ and label/")
    p.add_argument("--regions", nargs="*", default=None, help="Region ids (strings). If omitted, use all.")
    p.add_argument(
        "--months",
        nargs="*",
        type=int,
        default=[11, 12, 1, 2, 3, 4, 5, 6, 7],
        help="Months to include",
    )
    p.add_argument(
        "--train-fraction",
        type=float,
        default=0.01,
        help="Fraction of ALL tiles for training",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Fraction of REMAINING tiles for testing",
    )
    p.add_argument(
        "--pixels-per-tile",
        type=int,
        default=4096,
        help="Max valid pixels sampled per tile",
    )
    p.add_argument(
        "--balance-pixels",
        action="store_true",
        help="Class-balance pixel sampling within tiles",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--svm-kernel", default="rbf", help="SVM kernel (rbf, linear, poly, sigmoid)")
    p.add_argument("--svm-C", type=float, default=1.0, help="SVM C parameter")
    p.add_argument(
        "--svm-gamma",
        default="scale",
        help="SVM gamma (scale, auto, or float)",
    )
    p.add_argument(
        "--save-model",
        default=None,
        help="Optional path to save the trained model (.joblib)",
    )
    p.add_argument(
        "--use-meta-stats",
        action="store_true",
        help="Use precomputed meta statistics for normalization (./meta/)",
    )

    a = p.parse_args()
    months = tuple(int(m) for m in a.months)
    try:
        svm_gamma: str | float = float(a.svm_gamma)
    except ValueError:
        svm_gamma = str(a.svm_gamma)
    return Config(
        root=str(a.root),
        year=str(a.year),
        regions=[str(r) for r in a.regions] if a.regions else None,
        months=months,
        train_fraction=float(a.train_fraction),
        test_fraction=float(a.test_fraction),
        pixels_per_tile=int(a.pixels_per_tile),
        balance_pixels=bool(a.balance_pixels),
        seed=int(a.seed),
        svm_kernel=str(a.svm_kernel),
        svm_C=float(a.svm_C),
        svm_gamma=svm_gamma,
        save_model=str(a.save_model) if a.save_model else None,
        use_meta_stats=bool(a.use_meta_stats),
    )


if __name__ == "__main__":
    cfg = parse_args()
    res = train_and_eval(cfg)
    print(res)
