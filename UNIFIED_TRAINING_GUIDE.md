# Unified Training Module - User Guide

## Overview

All training code for the 4 sklearn-compatible models (XGBoost, HistGradientBoosting, RandomForest, SVM) has been consolidated into a single unified interface at `server/train/sklearn_train.py`.

## Key Benefits

✅ **Single Import**: One module instead of 4 separate scripts  
✅ **Unified Config**: Same `TrainConfig` dataclass for all models  
✅ **Factory Pattern**: Model creation via `create_model(config)`  
✅ **Meta Stats Support**: Built-in support for per-month normalization  
✅ **API Integration**: Works seamlessly with FastAPI endpoints  
✅ **Backward Compatible**: CLI wrapper maintains old script interfaces  

## Architecture

```
server/train/sklearn_train.py       # Core training logic
├── TrainConfig                      # Universal config dataclass
├── train_sklearn_model(cfg)         # Main training function
├── create_model(cfg)                # Model factory (XGB/HistGB/RF/SVM)
└── Convenience functions:
    ├── train_xgboost()
    ├── train_histgb()
    ├── train_random_forest()
    └── train_svm()

server/model/sklearn_models.py      # Model wrappers
├── SklearnWheatModel               # AbstractModel implementation
├── predict_pixel(array) → (n,)     # Pixel-level prediction
├── predict_patch(array) → (H,W)    # Patch-level prediction
└── Convenience classes:
    ├── XGBoostWheatModel
    ├── HistGBWheatModel
    ├── RandomForestWheatModel
    └── SVMWheatModel

server/train/train_unified.py       # CLI wrapper (backward compatible)
```

## Usage Methods

### Method 1: Unified CLI (Recommended)

```bash
# XGBoost with meta stats
python server/train/train_unified.py \
  --model xgboost \
  --root "C:\Users\Administrator\Desktop\preprocessed_data" \
  --year 2020 \
  --train-fraction 0.01 \
  --use-meta-stats \
  --save-path "runs/xgb_unified.joblib"

# HistGradientBoosting without meta stats
python server/train/train_unified.py \
  --model histgb \
  --root "C:\Users\Administrator\Desktop\preprocessed_data" \
  --year 2020 \
  --train-fraction 0.01 \
  --max-iter 200

# Random Forest
python server/train/train_unified.py \
  --model random_forest \
  --root "C:\Users\Administrator\Desktop\preprocessed_data" \
  --year 2020 \
  --train-fraction 0.01 \
  --rf-estimators 100

# SVM
python server/train/train_unified.py \
  --model svm \
  --root "C:\Users\Administrator\Desktop\preprocessed_data" \
  --year 2020 \
  --train-fraction 0.005
```

**Available Models**: `xgboost`, `histgb`, `random_forest`, `svm`

### Method 2: Python API

```python
from server.train.sklearn_train import TrainConfig, train_sklearn_model

# Configure training
config = TrainConfig(
    model_type="xgboost",
    root=r"C:\Users\Administrator\Desktop\preprocessed_data",
    year="2020",
    months=(11, 12, 1, 2, 3, 4, 5, 6, 7),
    train_fraction=0.01,
    pixels_per_tile=4096,
    balance_pixels=True,
    use_meta_stats=True,
    meta_dir="./meta",
    n_estimators=100,
    max_depth=8,
    save_path="runs/xgb_custom.joblib"
)

# Train model
results = train_sklearn_model(config)

# Access results
print(f"F1 Score: {results['f1']:.4f}")
print(f"IoU: {results['iou']:.4f}")
print(f"Model: {results['model_path']}")
```

### Method 3: FastAPI Server

**Start server:**
```bash
uvicorn server.server.app:app --reload
```

**Submit training job (Python):**
```python
import requests

response = requests.post("http://localhost:8000/train", json={
    "job_name": "my_xgboost_experiment",
    "algorithm": "xgboost",
    "dataset": {
        "root": r"C:\Users\Administrator\Desktop\preprocessed_data",
        "year": "2020",
        "train_fraction": 0.01,
        "use_meta_stats": True,
        "meta_dir": "./meta"
    },
    "save_model": True,
    "model_params": {
        "n_estimators": 200,
        "max_depth": 10
    }
})

job_id = response.json()["job_id"]

# Check status
status = requests.get("http://localhost:8000/train/status", params={"id": job_id})
print(status.json())
```

**Or use the test script:**
```bash
python test_api_training.py
```

### Method 4: Model Wrapper Usage

```python
from server.model.sklearn_models import load_model
import numpy as np

# Load trained model
model = load_model("runs/xgb_2020.joblib")

# Pixel-level prediction (flat array)
pixels = np.random.randn(1000, 117)  # (n_pixels, n_features)
predictions = model.predict_pixel(pixels)  # (1000,) binary array

# Patch-level prediction (image array)
patch = np.random.randn(117, 64, 64)  # (n_features, H, W)
mask = model.predict_patch(patch)  # (64, 64) binary mask

# Evaluate on dataset
from server.dataset.PatchDataset import WheatTilesDataset
dataset = WheatTilesDataset(
    root_preprocessed=r"C:\Users\Administrator\Desktop\preprocessed_data",
    year="2020",
    month_order=(11, 12, 1, 2, 3, 4, 5, 6, 7)
)
metrics = model.val_pixel_dataset(dataset, max_pixels=100000)
print(f"Test F1: {metrics['f1']:.4f}")
```

## Configuration Options

### TrainConfig Parameters

**Dataset:**
- `root`: Path to preprocessed tile data
- `year`: Year to train on (e.g., "2020")
- `regions`: List of regions or None for all
- `months`: Tuple of months (e.g., `(11, 12, 1, 2, 3, 4, 5, 6, 7)`)
- `train_fraction`: Fraction of tiles for training (0.0-1.0)
- `test_fraction`: Fraction of pixels per tile for testing
- `pixels_per_tile`: Number of pixels to sample per tile
- `balance_pixels`: Whether to balance wheat/non-wheat pixels
- `seed`: Random seed

**Normalization:**
- `use_meta_stats`: Use precomputed mean/std (recommended)
- `meta_dir`: Path to folder with `.npz` stats files

**Model Type:**
- `model_type`: `"xgboost"`, `"hist_gradient_boosting"`, `"random_forest"`, or `"svm"`

**XGBoost Hyperparameters:**
- `n_estimators`: Number of boosting rounds (default: 400)
- `max_depth`: Maximum tree depth (default: 8)
- `learning_rate`: Learning rate (default: 0.05)
- `subsample`: Row subsampling (default: 0.8)
- `colsample_bytree`: Column subsampling (default: 0.8)

**HistGradientBoosting Hyperparameters:**
- `max_iter`: Number of boosting iterations (default: 400)
- `max_depth`: Maximum tree depth (default: 8)
- `learning_rate`: Learning rate (default: 0.05)
- `l2_regularization`: L2 regularization (default: 0.0)

**RandomForest Hyperparameters:**
- `rf_estimators`: Number of trees (default: 200)
- `rf_max_depth`: Maximum tree depth (default: None)

**SVM Hyperparameters:**
- `svm_kernel`: Kernel type (default: "rbf")
- `svm_C`: Regularization parameter (default: 1.0)
- `svm_gamma`: Kernel coefficient (default: "scale")

**Output:**
- `save_path`: Where to save trained model (`.joblib`)

## Testing & Validation

### Quick Test (2-3 minutes)
```bash
python server/train/train_unified.py --model xgboost --root "C:\Users\Administrator\Desktop\preprocessed_data" --year 2020 --train-fraction 0.01 --use-meta-stats
```

### Meta Stats Comparison
```bash
# WITH meta stats
python server/train/train_unified.py --model xgboost --root "C:\Users\Administrator\Desktop\preprocessed_data" --year 2020 --train-fraction 0.01 --use-meta-stats --save-path runs/xgb_meta.joblib

# WITHOUT meta stats
python server/train/train_unified.py --model xgboost --root "C:\Users\Administrator\Desktop\preprocessed_data" --year 2020 --train-fraction 0.01 --save-path runs/xgb_no_meta.joblib

# Expected: +6-8% F1/IoU improvement with meta stats
```

### Full Training (Production)
```bash
python server/train/train_unified.py --model xgboost --root "C:\Users\Administrator\Desktop\preprocessed_data" --year 2020 --train-fraction 0.5 --use-meta-stats --n-estimators 400 --save-path runs/xgb_production.joblib
```

### API Server Test
```bash
# Terminal 1: Start server
uvicorn server.server.app:app --reload

# Terminal 2: Run test
python test_api_training.py
```

## Migration from Old Scripts

**Old way:**
```bash
python scripts/train_xgboost.py --root ... --year 2020 --n-estimators 400
python scripts/train_histgb.py --root ... --year 2020 --max-iter 400
python scripts/train_rf_baseline.py --root ... --year 2020 --rf-estimators 200
python scripts/train_svm_baseline.py --root ... --year 2020 --svm-C 1.0
```

**New way:**
```bash
python server/train/train_unified.py --model xgboost --root ... --year 2020 --n-estimators 400
python server/train/train_unified.py --model histgb --root ... --year 2020 --max-iter 400
python server/train/train_unified.py --model random_forest --root ... --year 2020 --rf-estimators 200
python server/train/train_unified.py --model svm --root ... --year 2020 --svm-C 1.0
```

**Or from Python:**
```python
# Old way
from server.train.train_xgboost import Config, train_and_eval
cfg = Config(root=..., year="2020", n_estimators=400)
result = train_and_eval(cfg)

# New way
from server.train.sklearn_train import train_xgboost
result = train_xgboost(root=..., year="2020", n_estimators=400)
```

## Expected Performance

| Model | F1 Score (no meta) | F1 Score (with meta) | Improvement |
|-------|-------------------|---------------------|-------------|
| XGBoost | 0.7552 | 0.8153 | +8.0% |
| HistGB | ~0.75 | ~0.82 | +6-8% |
| Random Forest | ~0.70 | ~0.76 | +6-8% |
| SVM | ~0.68 | ~0.74 | +6-8% |

*Based on train_fraction=0.01, year=2020*

## Troubleshooting

**Import Error:**
```bash
python -c "from server.train.sklearn_train import TrainConfig; print('OK')"
```

**Model Loading:**
```python
from server.model.sklearn_models import load_model
model = load_model("runs/xgb_2020.joblib")
print(type(model))  # Should be SklearnWheatModel
```

**API Connection:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok"}
```
