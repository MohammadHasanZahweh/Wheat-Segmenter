"""
Consolidated training module for sklearn-based wheat segmentation models.

Supports:
- XGBoost (XGBClassifier)
- HistGradientBoosting (HistGradientBoostingClassifier)
- RandomForest (RandomForestClassifier)
- SVM (SVC with StandardScaler pipeline)

All models use the same dataset interface (WheatTilesDataset + StratifiedRandomSubset)
and share common training/evaluation logic.
"""

from __future__ import annotations

import sys
import os

# Enable unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import sys
import os

import numpy as np
import joblib
from torch.utils.data import Subset

# Ensure project root is on sys.path for ml_utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server.dataset.PatchDataset import WheatTilesDataset, StratifiedRandomSubset
from ml_utils import build_xy_from_tiles, f1_iou


ModelType = Literal["xgboost", "histgb", "random_forest", "svm"]


@dataclass
class TrainConfig:
    """Universal configuration for sklearn-based wheat segmentation training."""
    # Data config
    root: str
    year: str
    regions: list[str] | None = None
    months: tuple[int, ...] = (11, 12, 1, 2, 3, 4, 5, 6, 7)
    
    # Sampling config
    train_fraction: float = 0.01
    test_fraction: float = 0.25
    pixels_per_tile: int = 4096
    balance_pixels: bool = False
    seed: int = 42
    
    # Normalization config
    use_meta_stats: bool = False
    meta_dir: str = "./meta"
    
    # Model selection
    model_type: ModelType = "xgboost"
    
    # XGBoost hyperparameters
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # HistGradientBoosting hyperparameters
    hgb_max_depth: int | None = 8
    hgb_max_iter: int = 400
    hgb_learning_rate: float = 0.05
    hgb_l2_regularization: float = 0.0
    
    # RandomForest hyperparameters
    rf_n_estimators: int = 200
    rf_max_depth: int | None = None
    
    # SVM hyperparameters
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    svm_gamma: str | float = "scale"
    
    # Output config
    save_model: str | None = None


def load_dataset(cfg: TrainConfig) -> WheatTilesDataset:
    """Load WheatTilesDataset with configuration."""
    return WheatTilesDataset(
        root_preprocessed=cfg.root,
        year=cfg.year,
        regions=cfg.regions,
        month_order=cfg.months,
        temporal_layout=True,
        normalize=True,
        band_stats='auto' if cfg.use_meta_stats else None,
        meta_dir=cfg.meta_dir if cfg.use_meta_stats else None,
        require_complete=True,
        target_bands=None,
        target_size=(64, 64),
        size_policy="pad",
        probe_limit=12,
    )


def create_model(cfg: TrainConfig):
    """Factory function to create sklearn-compatible model based on config."""
    if cfg.model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            tree_method="hist",
            objective="binary:logistic",
            n_jobs=-1,
            random_state=cfg.seed,
            eval_metric="logloss",
        )
    
    elif cfg.model_type == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_depth=cfg.hgb_max_depth,
            max_iter=cfg.hgb_max_iter,
            learning_rate=cfg.hgb_learning_rate,
            l2_regularization=cfg.hgb_l2_regularization,
            loss="log_loss",
            random_state=cfg.seed,
        )
    
    elif cfg.model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            n_jobs=-1,
            random_state=cfg.seed,
            class_weight=None,
        )
    
    elif cfg.model_type == "svm":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel=cfg.svm_kernel,
                C=cfg.svm_C,
                gamma=cfg.svm_gamma,
                probability=True,
                random_state=cfg.seed,
            ),
        )
    
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")


def sample_tiles(ds: WheatTilesDataset, cfg: TrainConfig) -> tuple[list[int], list[int]]:
    """Sample train and test tiles using stratified sampling."""
    print(f"Sampling TRAIN ~{cfg.train_fraction*100:.2f}% of tiles (stratified)...")
    train_tiles = list(iter(StratifiedRandomSubset(ds, fraction=cfg.train_fraction, n_bins=5, seed=cfg.seed)))
    print(f"Train tiles: {len(train_tiles)}")
    
    if len(train_tiles) == 0:
        raise RuntimeError("Train sampler returned 0 tiles. Increase train_fraction or check data.")
    
    all_ids = set(range(len(ds)))
    remaining = sorted(all_ids.difference(set(train_tiles)))
    
    if len(remaining) == 0:
        raise RuntimeError("No remaining tiles to sample test set from. Lower train_fraction.")
    
    rem_subset = Subset(ds, remaining)
    print(f"Sampling TEST ~{cfg.test_fraction*100:.2f}% of remaining tiles (stratified)...")
    test_sampler = StratifiedRandomSubset(rem_subset, fraction=cfg.test_fraction, n_bins=5, seed=cfg.seed + 7)
    val_tiles = [remaining[i] for i in iter(test_sampler)]
    print(f"Test tiles: {len(val_tiles)} (sampled from {len(remaining)} remaining)")
    
    return train_tiles, val_tiles


def train_sklearn_model(cfg: TrainConfig) -> dict[str, Any]:
    """
    Main training function for sklearn-based models.
    
    Returns:
        Dictionary with training results including metrics and paths.
    """
    print(f"[INFO] Training {cfg.model_type} model", flush=True)
    print("=" * 60, flush=True)

    # Load dataset
    print("Loading dataset...", flush=True)
    ds = load_dataset(cfg)
    print(f"Indexed tiles: {len(ds)}")
    
    # Sample tiles
    train_tiles, val_tiles = sample_tiles(ds, cfg)
    
    # Build training pixel matrix
    print("Building training pixel matrix...")
    X_train, y_train = build_xy_from_tiles(ds, train_tiles, cfg.pixels_per_tile, cfg.balance_pixels, cfg.seed)
    print(f"Train pixels: {len(y_train)} | features: {X_train.shape[1] if X_train.size else 0}")
    
    if len(y_train) == 0:
        raise RuntimeError("No training pixels extracted. Increase pixels_per_tile or adjust data.")
    
    # Create and train model
    print(f"Training {cfg.model_type}...")
    model = create_model(cfg)
    model.fit(X_train, y_train)
    
    # Prepare results
    results = {
        "status": "completed",
        "model_type": cfg.model_type,
        "train_tiles": len(train_tiles),
        "test_tiles": len(val_tiles),
        "train_pixels": len(y_train),
        "features": X_train.shape[1] if X_train.size else 0,
    }
    
    # Evaluate on test set
    if val_tiles:
        print("Building test pixel matrix...")
        X_val, y_val = build_xy_from_tiles(ds, val_tiles, cfg.pixels_per_tile, False, cfg.seed + 1)
        print(f"Test pixels: {len(y_val)}")
        
        if len(y_val) > 0:
            # Predict
            if hasattr(model, "predict_proba"):
                y_pred = (model.predict_proba(X_val)[:, 1] >= 0.5).astype(np.uint8)
            elif hasattr(model, "decision_function"):
                y_pred = (model.decision_function(X_val) >= 0.0).astype(np.uint8)
            else:
                y_pred = model.predict(X_val)
            
            # Metrics
            f1, iou = f1_iou(y_val, y_pred)
            pos_rate = float(y_val.mean()) if len(y_val) > 0 else 0.0
            
            results.update({
                "test_pixels": len(y_val),
                "f1": f1,
                "iou": iou,
                "positive_rate": pos_rate,
            })
            
            print(f"Test: F1={f1:.4f} | IoU={iou:.4f} | PosRate={pos_rate:.3f}")
        else:
            print("Test set had 0 pixels after filtering.")
            results["test_pixels"] = 0
    else:
        print("No test tiles; reporting train metrics only.")
        if hasattr(model, "predict_proba"):
            y_pred_tr = (model.predict_proba(X_train)[:, 1] >= 0.5).astype(np.uint8)
        elif hasattr(model, "decision_function"):
            y_pred_tr = (model.decision_function(X_train) >= 0.0).astype(np.uint8)
        else:
            y_pred_tr = model.predict(X_train)
        
        f1, iou = f1_iou(y_train, y_pred_tr)
        pos_rate = float(y_train.mean())
        
        results.update({
            "train_f1": f1,
            "train_iou": iou,
            "train_positive_rate": pos_rate,
        })
        
        print(f"Train: F1={f1:.4f} | IoU={iou:.4f} | PosRate={pos_rate:.3f}", flush=True)
    
    # Save model
    if cfg.save_model:
        save_path = Path(cfg.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        results["model_path"] = str(save_path)
        print(f"Saved model to {save_path}", flush=True)
    
    return results


# Convenience functions for each model type
def train_xgboost(cfg: TrainConfig) -> dict[str, Any]:
    """Train XGBoost model."""
    cfg.model_type = "xgboost"
    return train_sklearn_model(cfg)


def train_histgb(cfg: TrainConfig) -> dict[str, Any]:
    """Train HistGradientBoosting model."""
    cfg.model_type = "histgb"
    return train_sklearn_model(cfg)


def train_random_forest(cfg: TrainConfig) -> dict[str, Any]:
    """Train RandomForest model."""
    cfg.model_type = "random_forest"
    return train_sklearn_model(cfg)


def train_svm(cfg: TrainConfig) -> dict[str, Any]:
    """Train SVM model."""
    cfg.model_type = "svm"
    return train_sklearn_model(cfg)


if __name__ == "__main__":
    # Example usage
    config = TrainConfig(
        root="C:/Users/Administrator/Desktop/preprocessed_data",
        year="2020",
        train_fraction=0.01,
        pixels_per_tile=2048,
        balance_pixels=True,
        use_meta_stats=True,
        model_type="xgboost",
        xgb_n_estimators=100,
        save_model="./runs/test_consolidated_xgb.joblib"
    )
    
    results = train_sklearn_model(config)
    print("\nFinal Results:")
    print(results)
