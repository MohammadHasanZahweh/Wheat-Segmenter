"""
Sklearn-based model wrappers for wheat segmentation.

Provides a unified interface for XGBoost, HistGradientBoosting, RandomForest, and SVM
models that can be used with the AbstractModel interface for API/Streamlit integration.
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from typing import Literal, Any, Dict, Sequence, Optional

from server.model.base_model import AbstractModel


ModelType = Literal["xgboost", "histgb", "random_forest", "svm"]


def load_meta_stats(meta_dir: Path, year: str, months: Sequence[int]) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load precomputed mean/std statistics from meta directory.
    
    Expected structure:
        meta/<YEAR>_<MONTH>.npz containing 'mean' and 'std' arrays
    
    Args:
        meta_dir: Path to meta directory
        year: Year string (e.g., '2020')
        months: Sequence of month integers to load
    
    Returns:
        Dict mapping month -> {'mean': np.ndarray, 'std': np.ndarray}
    """
    meta_stats = {}
    for month in months:
        npz_path = meta_dir / f"{year}_{month}.npz"
        if not npz_path.exists():
            print(f"[WARN] Meta stats not found: {npz_path}, skipping month {month}")
            continue
        data = np.load(npz_path)
        if 'mean' not in data or 'std' not in data:
            print(f"[WARN] Meta file {npz_path} missing 'mean' or 'std', skipping")
            continue
        meta_stats[month] = {
            'mean': data['mean'].astype(np.float32),
            'std': data['std'].astype(np.float32)
        }
    return meta_stats


class SklearnWheatModel(AbstractModel):
    """
    Wrapper for sklearn-compatible wheat segmentation models.
    
    Wraps XGBoost, HistGradientBoosting, RandomForest, or SVM classifiers
    to work with the AbstractModel interface.
    
    Supports normalization for inference using:
    - Meta statistics (per-month mean/std) 
    - Per-tile min-max normalization
    """
    
    def __init__(self, model_type: ModelType = "xgboost", 
                 year: str = "2020",
                 months: Sequence[int] = (11, 12, 1, 2, 3, 4, 5, 6, 7),
                 meta_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize sklearn model.
        
        Args:
            model_type: Type of model ("xgboost", "histgb", "random_forest", "svm")
            year: Year for loading meta statistics
            months: Month sequence for temporal data
            meta_dir: Directory containing meta statistics (if None, uses per-tile normalization)
            **kwargs: Model-specific hyperparameters
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.num_classes = 2  # Binary classification: wheat vs non-wheat
        
        # Normalization configuration
        self.year = year
        self.months = months
        self.meta_dir = meta_dir
        self.meta_stats = None
        
        # Load meta statistics if provided
        if meta_dir:
            meta_path = Path(meta_dir) / "wheat"
            if meta_path.exists():
                self.meta_stats = load_meta_stats(meta_path, self.year, self.months)
                if self.meta_stats:
                    print(f"[INFO] Loaded meta stats for {len(self.meta_stats)} months from {meta_path}")
            else:
                print(f"[WARN] meta_dir specified but not found: {meta_path}")
        
    def _create_model(self, **kwargs):
        """Create the appropriate sklearn model."""
        if self.model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 400),
                max_depth=kwargs.get('max_depth', 8),
                learning_rate=kwargs.get('learning_rate', 0.05),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                tree_method="hist",
                objective="binary:logistic",
                n_jobs=-1,
                random_state=kwargs.get('random_state', 42),
                eval_metric="logloss",
            )
        
        elif self.model_type == "histgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_depth=kwargs.get('max_depth', 8),
                max_iter=kwargs.get('max_iter', 400),
                learning_rate=kwargs.get('learning_rate', 0.05),
                l2_regularization=kwargs.get('l2_regularization', 0.0),
                loss="log_loss",
                random_state=kwargs.get('random_state', 42),
            )
        
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', None),
                n_jobs=-1,
                random_state=kwargs.get('random_state', 42),
            )
        
        elif self.model_type == "svm":
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            return make_pipeline(
                StandardScaler(),
                SVC(
                    kernel=kwargs.get('kernel', 'rbf'),
                    C=kwargs.get('C', 1.0),
                    gamma=kwargs.get('gamma', 'scale'),
                    probability=True,
                    random_state=kwargs.get('random_state', 42),
                ),
            )
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def save(self, path: str):
        """Save model to disk using joblib."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model along with normalization config
        save_data = {
            'model': self.model,
            'model_type': self.model_type,
            'year': self.year,
            'months': self.months,
            'meta_dir': self.meta_dir,
            'has_meta_stats': self.meta_stats is not None
        }
        joblib.dump(save_data, save_path)
        print(f"Model saved to {save_path}")
    
    def load(self, path: str):
        """Load model from disk."""
        loaded = joblib.load(path)
        
        # Handle both old format (just model) and new format (dict with metadata)
        if isinstance(loaded, dict) and 'model' in loaded:
            self.model = loaded['model']
            self.model_type = loaded.get('model_type', self.model_type)
            self.year = loaded.get('year', self.year)
            self.months = loaded.get('months', self.months)
            self.meta_dir = loaded.get('meta_dir', self.meta_dir)
            
            # Reload meta stats if needed
            if loaded.get('has_meta_stats') and self.meta_dir:
                meta_path = Path(self.meta_dir) / "wheat"
                if meta_path.exists():
                    self.meta_stats = load_meta_stats(meta_path, self.year, self.months)
        else:
            # Old format - just the model
            self.model = loaded
        
        print(f"Model loaded from {path}")
    
    def _normalize_patch(self, array: np.ndarray) -> np.ndarray:
        """
        Normalize a patch using global mean/std computed during training.
        
        Args:
            array: Shape (T, B, H, W) - temporal, bands, height, width
        
        Returns:
            Normalized array with same shape
        """
        # Use global mean/std (matching training normalization)
        if self.mean.size > 1 and self.std.size > 1:
            # Reshape mean and std to (T, B, 1, 1) for broadcasting
            mean_reshaped = self.mean.reshape(self.mean.shape[0], self.mean.shape[1], 1, 1)
            std_reshaped = self.std.reshape(self.std.shape[0], self.std.shape[1], 1, 1)
            return (array - mean_reshaped) / std_reshaped
        else:
            # No normalization if mean/std not computed
            return array
    
    def predict_pixel(self, array: np.ndarray) -> np.ndarray:
        """
        Predict wheat presence for pixel features.
        
        Args:
            array: Shape (n, T*B) where T=months, B=bands
                   Flattened temporal-spectral features per pixel
        
        Returns:
            Shape (n,) binary predictions (0=non-wheat, 1=wheat)
        """
        if array.ndim == 3:  # (n, T, B) -> flatten to (n, T*B)
            n, T, B = array.shape
            array = array.reshape(n, T * B)
        
        # Predict probabilities
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(array)[:, 1]
            return (proba >= 0.5).astype(np.uint8)
        elif hasattr(self.model, 'decision_function'):
            decision = self.model.decision_function(array)
            return (decision >= 0.0).astype(np.uint8)
        else:
            return self.model.predict(array).astype(np.uint8)
    
    def predict_patch(self, array: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Predict wheat for entire patch/tile.
        
        Args:
            array: Shape (T, B, H, W) - temporal, bands, height, width
            normalize: Whether to normalize the input (default True)
        
        Returns:
            Shape (H, W) binary prediction mask
        """
        T, B, H, W = array.shape
        
        # Normalize if requested
        if normalize:
            array = self._normalize_patch(array)
        
        # Reshape to (H*W, T*B)
        pixels = array.reshape(T * B, H * W).T  # (H*W, T*B)
        
        # Predict per pixel
        predictions = self.predict_pixel(pixels)
        
        # Reshape back to (H, W)
        return predictions.reshape(H, W)
    
    def fit_pixel(self, dataset):
        """
        Train on pixel-level dataset.
        
        Args:
            dataset: Should yield (features, labels) where features are (T*B,) per pixel
        """
        # Collect all data from dataset
        X_list = []
        y_list = []
        
        for batch in dataset:
            if isinstance(batch, dict):
                X_list.append(batch['features'])
                y_list.append(batch['labels'])
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                X_list.append(batch[0])
                y_list.append(batch[1])
            else:
                raise ValueError("Dataset must yield (features, labels) tuples or dicts")
        
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        print(f"Training {self.model_type} on {len(y)} pixels with {X.shape[1]} features...")
        self.model.fit(X, y)
        print("Training complete!")
    
    def fit_patch(self, dataset):
        """Train on patch-level dataset (not typically used for sklearn models)."""
        raise NotImplementedError("Patch-based training not implemented for sklearn models")
    
    def val_pixel_dataset(self, dataset, prefix: str = "") -> dict[str, Any]:
        """
        Evaluate on pixel-level dataset.
        
        Args:
            dataset: Validation dataset
            prefix: Prefix for metric keys (e.g., "val_", "test_")
        
        Returns:
            Dictionary with metrics
        """
        # Collect all data
        X_list = []
        y_list = []
        
        for batch in dataset:
            if isinstance(batch, dict):
                X_list.append(batch['features'])
                y_list.append(batch['labels'])
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                X_list.append(batch[0])
                y_list.append(batch[1])
        
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        # Predict
        y_pred = self.predict_pixel(X)
        
        # Calculate metrics
        from .ml_utils import f1_iou
        f1, iou = f1_iou(y, y_pred)
        
        accuracy = (y == y_pred).mean()
        
        return {
            f"{prefix}accuracy": float(accuracy),
            f"{prefix}f1": float(f1),
            f"{prefix}iou": float(iou),
            f"{prefix}samples": len(y),
        }
    
    def eval_patch_dataset(self, dataset):
        """Evaluate on patch dataset."""
        raise NotImplementedError("Patch evaluation not implemented yet")
    
    def eval_pixel_dataset(self, dataset):
        """Alias for val_pixel_dataset."""
        return self.val_pixel_dataset(dataset, prefix="eval_")


def load_model(path: str, meta_dir: Optional[str] = None, 
                year: str = "2020", 
                months: Sequence[int] = (11, 12, 1, 2, 3, 4, 5, 6, 7)) -> SklearnWheatModel:
    """
    Load a trained sklearn wheat model from disk.
    
    Args:
        path: Path to .joblib file
        meta_dir: Optional path to meta statistics directory for normalization
        year: Year for meta statistics (if meta_dir provided)
        months: Month sequence for meta statistics (if meta_dir provided)
    
    Returns:
        SklearnWheatModel instance with loaded model
    """
    loaded = joblib.load(path)
    
    # Handle both old format (just model) and new format (dict with metadata)
    if isinstance(loaded, dict) and 'model' in loaded:
        # New format with metadata
        raw_model = loaded['model']
        model_type = loaded.get('model_type', 'xgboost')
        saved_year = loaded.get('year', year)
        saved_months = loaded.get('months', months)
        saved_meta_dir = loaded.get('meta_dir', meta_dir)
        
        # Create wrapper with saved configuration
        wrapper = SklearnWheatModel(
            model_type=model_type,
            year=saved_year,
            months=saved_months,
            meta_dir=saved_meta_dir or meta_dir  # Allow override
        )
        wrapper.model = raw_model
        
        # Restore normalization statistics
        wrapper.mean = np.array(loaded.get('mean', [0]), dtype=np.float32)
        wrapper.std = np.array(loaded.get('std', [1]), dtype=np.float32)
    else:
        # Old format - just the model, detect type
        raw_model = loaded
        model_type = "xgboost"  # Default
        if hasattr(raw_model, '__class__'):
            class_name = raw_model.__class__.__name__
            if 'HistGradient' in class_name:
                model_type = "histgb"
            elif 'RandomForest' in class_name:
                model_type = "random_forest"
            elif 'Pipeline' in class_name or 'SVC' in class_name:
                model_type = "svm"
        
        # Create wrapper with provided configuration
        wrapper = SklearnWheatModel(
            model_type=model_type,
            year=year,
            months=months,
            meta_dir=meta_dir
        )
        wrapper.model = raw_model
    
    return wrapper


# Convenience classes for each model type
class XGBoostWheatModel(SklearnWheatModel):
    """XGBoost wheat segmentation model."""
    def __init__(self, **kwargs):
        super().__init__(model_type="xgboost", **kwargs)


class HistGBWheatModel(SklearnWheatModel):
    """HistGradientBoosting wheat segmentation model."""
    def __init__(self, **kwargs):
        super().__init__(model_type="histgb", **kwargs)


class RandomForestWheatModel(SklearnWheatModel):
    """RandomForest wheat segmentation model."""
    def __init__(self, **kwargs):
        super().__init__(model_type="random_forest", **kwargs)


class SVMWheatModel(SklearnWheatModel):
    """SVM wheat segmentation model."""
    def __init__(self, **kwargs):
        super().__init__(model_type="svm", **kwargs)
