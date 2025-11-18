# Training Scripts - Usage Guide

This directory contains all training scripts for wheat segmentation using classical ML models. All scripts use the consolidated `WheatTilesDataset` and `StratifiedRandomSubset` sampler from `server/dataset/PatchDataset.py`.

## Available Training Scripts

1. **XGBoost Classifier** (`train_xgboost.py`)
2. **Histogram Gradient Boosting** (`train_histgb.py`)
3. **Random Forest** (`train_rf_baseline.py`)
4. **Support Vector Machine** (`train_svm_baseline.py`)

### Full Training Profiles
For production-scale (non-quick) configurations see `FULL_TRAINING_COMMANDS.md` in this directory. It lists Moderate, Extensive, and Max coverage profiles plus sequential batch commands and resource guidance.

---

## Prerequisites

Ensure you have:
- Preprocessed data in the structure: `data/<YEAR>/<REGION>/<MONTH>/<TILE_ID>.tif`
- Labels in the structure: `label/<YEAR>/<REGION>/<TILE_ID>.tif`
- Required Python packages installed (see `requirements.txt`)

**Important**: The `--root` parameter should point to the directory that **contains** the `data/` and `label/` folders. 

### Finding Your Data Path

If your data structure is:
```
/path/to/preprocessed/
├── data/
│   └── 2020/
│       ├── 0/
│       ├── 1/
│       └── ...
└── label/
    └── 2020/
        ├── 0/
        ├── 1/
        └── ...
```

Then use `--root /path/to/preprocessed`

**Common locations:**
- If data is in project root: `--root .`
- If data is in a subdirectory: `--root ./preprocessed` or `--root "C:/path/to/data"`
- If data is on another drive: `--root "D:/wheat_data"`

**Example**: If your preprocessed data is at `C:\Users\Administrator\Desktop\preprocessed_data`, use:
```bash
--root "C:\Users\Administrator\Desktop\preprocessed_data"
```

**To find your data**, run:
```bash
python find_data_path.py
```

---

## Common Parameters

All training scripts share these common parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--root` | Path to preprocessed root containing `data/` and `label/` | **Required** |
| `--year` | Year subfolder under `data/` and `label/` | **Required** |
| `--regions` | Region IDs to use (space-separated). If omitted, uses all regions | All regions |
| `--months` | Months to include (space-separated integers) | `11 12 1 2 3 4 5 6 7` |
| `--train-fraction` | Fraction of ALL tiles for training (stratified sampling) | `0.01` (1%) |
| `--test-fraction` | Fraction of REMAINING tiles for testing (stratified sampling) | `0.25` (25%) |
| `--pixels-per-tile` | Max valid pixels sampled per tile | `4096` |
| `--balance-pixels` | Class-balance pixel sampling within tiles | `False` |
| `--seed` | Random seed for reproducibility | `42` |
| `--save-model` | Path to save the trained model (.joblib file) | `None` |

---

## 1. XGBoost Classifier

### Description
Trains an XGBoost gradient boosting classifier for wheat segmentation. XGBoost is fast, efficient, and provides excellent performance on tabular data.

### Model-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n-estimators` | Number of boosting rounds | `400` |
| `--max-depth` | Maximum tree depth | `8` |
| `--learning-rate` | Learning rate (eta) | `0.05` |
| `--subsample` | Subsample ratio of training instances | `0.8` |
| `--colsample-bytree` | Subsample ratio of columns when constructing each tree | `0.8` |

### Test Commands

**Note**: Replace `<PATH_TO_DATA>` with the actual path to your preprocessed data directory that contains `data/` and `label/` folders.

#### Basic Test (Minimal - Quick)
```bash
python server/train/train_xgboost.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.001 \
  --test-fraction 0.1 \
  --pixels-per-tile 1000 \
  --n-estimators 50
```
**Explanation**: Uses only 0.1% of tiles for training, 10% of remaining for testing, fewer pixels and trees. Runs in a few minutes for quick validation.

**Example with actual path**:
```bash
# If your data is at C:\Users\Administrator\Desktop\preprocessed_data
python server/train/train_xgboost.py --root "C:\Users\Administrator\Desktop\preprocessed_data" --year 2020 --train-fraction 0.001 --n-estimators 50 --pixels-per-tile 1000
```

#### Standard Test (1% Training Data)
```bash
python server/train/train_xgboost.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.01 \
  --test-fraction 0.25 \
  --pixels-per-tile 4096 \
  --balance-pixels \
  --save-model ./runs/xgb_test.joblib
```
**Explanation**: Uses 1% of tiles for training (stratified), 25% of remaining for testing, with balanced pixel sampling. Saves model to `runs/xgb_test.joblib`.

#### Full Training (Recommended for Production)
```bash
python server/train/train_xgboost.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.01 \
  --test-fraction 0.25 \
  --pixels-per-tile 4096 \
  --balance-pixels \
  --n-estimators 400 \
  --max-depth 8 \
  --learning-rate 0.05 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --save-model ./runs/xgb_2020.joblib
```
**Explanation**: Full configuration with optimized hyperparameters. Expected runtime: 30-60 minutes depending on data size.

#### Specific Regions Only
```bash
python server/train/train_xgboost.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --regions 0 1 2 \
  --train-fraction 0.01 \
  --save-model ./runs/xgb_regions_012.joblib
```
**Explanation**: Trains only on regions 0, 1, and 2. Useful for regional models or testing.

---

## 2. Histogram Gradient Boosting

### Description
Trains a scikit-learn HistGradientBoostingClassifier. Similar to XGBoost but uses native scikit-learn implementation. Often faster on large datasets due to histogram-based splitting.

### Model-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max-depth` | Maximum tree depth (None for unlimited) | `8` |
| `--max-iter` | Number of boosting iterations | `400` |
| `--learning-rate` | Learning rate | `0.05` |
| `--l2-regularization` | L2 regularization strength | `0.0` |

### Test Commands

**Note**: Replace `<PATH_TO_DATA>` with the actual path to your preprocessed data directory.

#### Basic Test (Minimal - Quick)
```bash
python server/train/train_histgb.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.001 \
  --test-fraction 0.1 \
  --pixels-per-tile 1000 \
  --max-iter 50
```
**Explanation**: Quick test with minimal data. Runs in a few minutes.

#### Standard Test (1% Training Data)
```bash
python server/train/train_histgb.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.01 \
  --test-fraction 0.25 \
  --pixels-per-tile 4096 \
  --balance-pixels \
  --save-model ./runs/hgb_test.joblib
```
**Explanation**: Standard 1% training with balanced sampling. Good for development and testing.

#### Full Training (Recommended for Production)
```bash
python server/train/train_histgb.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.01 \
  --test-fraction 0.25 \
  --pixels-per-tile 4096 \
  --balance-pixels \
  --max-depth 8 \
  --max-iter 400 \
  --learning-rate 0.05 \
  --l2-regularization 0.0 \
  --save-model ./runs/hgb_2020.joblib
```
**Explanation**: Production-ready configuration. Expected runtime: 20-45 minutes.

#### With L2 Regularization
```bash
python server/train/train_histgb.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.01 \
  --pixels-per-tile 4096 \
  --balance-pixels \
  --l2-regularization 0.1 \
  --save-model ./runs/hgb_l2reg.joblib
```
**Explanation**: Adds L2 regularization to prevent overfitting. Useful when training on limited data.

---

## 3. Random Forest Classifier

### Description
Trains a Random Forest ensemble classifier. Robust and interpretable, though typically slower than gradient boosting methods.

### Model-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--rf-estimators` | Number of trees in the forest | `200` |
| `--rf-max-depth` | Maximum tree depth (None for unlimited) | `None` |

### Test Commands

**Note**: Replace `<PATH_TO_DATA>` with the actual path to your preprocessed data directory.

#### Basic Test (Minimal - Quick)
```bash
python server/train/train_rf_baseline.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.001 \
  --test-fraction 0.1 \
  --pixels-per-tile 1000 \
  --rf-estimators 50
```
**Explanation**: Quick test with 50 trees. Runs in a few minutes.

#### Standard Test (1% Training Data)
```bash
python server/train/train_rf_baseline.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.01 \
  --test-fraction 0.25 \
  --pixels-per-tile 4096 \
  --balance-pixels \
  --rf-estimators 200 \
  --save-model ./runs/rf_test.joblib
```
**Explanation**: Standard configuration with 200 trees. Expected runtime: 40-90 minutes.

#### Full Training (Recommended for Production)
```bash
python server/train/train_rf_baseline.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.01 \
  --test-fraction 0.25 \
  --pixels-per-tile 4096 \
  --balance-pixels \
  --rf-estimators 200 \
  --rf-max-depth 20 \
  --save-model ./runs/rf_2020.joblib
```
**Explanation**: Limited tree depth to prevent overfitting. Expected runtime: 40-90 minutes.

#### Large Forest (High Performance)
```bash
python server/train/train_rf_baseline.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.01 \
  --pixels-per-tile 4096 \
  --balance-pixels \
  --rf-estimators 500 \
  --save-model ./runs/rf_500trees.joblib
```
**Explanation**: Uses 500 trees for potentially better performance. Expected runtime: 1-2 hours.

---

## 4. Support Vector Machine (SVM)

### Description
Trains an SVM classifier with StandardScaler preprocessing. SVMs work well for smaller datasets but can be slow on large datasets. **Note**: This is the slowest trainer.

### Model-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--svm-kernel` | Kernel type: `rbf`, `linear`, `poly`, `sigmoid` | `rbf` |
| `--svm-C` | Regularization parameter | `1.0` |
| `--svm-gamma` | Kernel coefficient: `scale`, `auto`, or float value | `scale` |

### Test Commands

**Note**: Replace `<PATH_TO_DATA>` with the actual path to your preprocessed data directory.

#### Basic Test (Minimal - Quick)
```bash
python server/train/train_svm_baseline.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.0005 \
  --test-fraction 0.05 \
  --pixels-per-tile 500 \
  --svm-kernel rbf
```
**Explanation**: Very minimal test (0.05% training data, 500 pixels/tile). SVM is slow, so use minimal data for testing. Runs in 5-10 minutes.

#### Standard Test (Limited Data)
```bash
python server/train/train_svm_baseline.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.001 \
  --test-fraction 0.1 \
  --pixels-per-tile 2000 \
  --balance-pixels \
  --svm-kernel rbf \
  --svm-C 1.0 \
  --save-model ./runs/svm_test.joblib
```
**Explanation**: Uses 0.1% training data due to SVM's computational cost. Expected runtime: 30-60 minutes.

#### Linear Kernel (Faster)
```bash
python server/train/train_svm_baseline.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.001 \
  --test-fraction 0.1 \
  --pixels-per-tile 2000 \
  --balance-pixels \
  --svm-kernel linear \
  --svm-C 1.0 \
  --save-model ./runs/svm_linear.joblib
```
**Explanation**: Linear kernel is faster than RBF. Good for high-dimensional data. Expected runtime: 20-40 minutes.

#### RBF Kernel (Full - Use Carefully)
```bash
python server/train/train_svm_baseline.py \
  --root <PATH_TO_DATA> \
  --year 2020 \
  --train-fraction 0.01 \
  --test-fraction 0.25 \
  --pixels-per-tile 4096 \
  --balance-pixels \
  --svm-kernel rbf \
  --svm-C 1.0 \
  --svm-gamma scale \
  --save-model ./runs/svm_2020.joblib
```
**Explanation**: Full 1% training with RBF kernel. **WARNING**: This can take several hours (2-6 hours) depending on your hardware.

---

## Output Format

All training scripts produce similar output:

```
Loading dataset...
Indexed tiles: 12345
Sampling TRAIN ~1.00% of tiles (stratified)...
Train tiles: 123
Sampling TEST ~25.00% of remaining tiles (stratified)...
Test tiles: 789 (sampled from 12222 remaining)
Building training pixel matrix...
Train pixels: 502784 | features: 63
Training [Model Name]...
Building test pixel matrix...
Test pixels: 3223040
Test: F1=0.8542 | IoU=0.7456 | PosRate=0.234
Saved model to ./runs/model.joblib
```

### Metrics Explanation
- **F1**: F1 score (harmonic mean of precision and recall)
- **IoU**: Intersection over Union (Jaccard index)
- **PosRate**: Positive class rate (proportion of wheat pixels)

---

## Running from Module

You can also run scripts as Python modules:

```bash
# XGBoost
python -m server.train.train_xgboost --root <PATH_TO_DATA> --year 2020

# HistGB
python -m server.train.train_histgb --root <PATH_TO_DATA> --year 2020

# Random Forest
python -m server.train.train_rf_baseline --root <PATH_TO_DATA> --year 2020

# SVM
python -m server.train.train_svm_baseline --root <PATH_TO_DATA> --year 2020
```

---

## Performance Comparison

Based on typical runs with 1% training data (default settings):

| Model | Training Time | Inference Speed | F1 Score | Memory Usage |
|-------|---------------|-----------------|----------|--------------|
| **XGBoost** | 30-60 min | Fast | ~0.85 | Moderate |
| **HistGB** | 20-45 min | Fast | ~0.84 | Low |
| **Random Forest** | 40-90 min | Medium | ~0.82 | High |
| **SVM (RBF)** | 2-6 hours | Slow | ~0.83 | Very High |
| **SVM (Linear)** | 20-40 min | Medium | ~0.80 | Moderate |

**Recommendation**: Start with **HistGB** or **XGBoost** for best speed/performance trade-off.

---

## Troubleshooting

### Issue: "No tiles found"
**Solution**: Check that `--root` points to the correct directory containing `data/` and `label/` folders.

### Issue: "Train sampler returned 0 tiles"
**Solution**: Increase `--train-fraction` or ensure your data directory has tiles with all required months.

### Issue: "No training pixels extracted"
**Solution**: Increase `--pixels-per-tile` or check that your tiles have valid pixels.

### Issue: Out of memory
**Solution**: 
- Reduce `--pixels-per-tile`
- Reduce `--train-fraction`
- For RF: Reduce `--rf-estimators`
- For SVM: Use `--svm-kernel linear` instead of `rbf`

### Issue: Training too slow
**Solution**:
- Reduce `--train-fraction` and `--test-fraction`
- Reduce `--pixels-per-tile`
- For XGBoost: Reduce `--n-estimators`
- For HistGB: Reduce `--max-iter`
- For RF: Reduce `--rf-estimators`
- For SVM: Use minimal data or switch to faster models

---

## Examples: Complete Workflows

### Quick Validation Workflow
Test all models quickly to ensure everything works:

```bash
# Test XGBoost (5 min)
python server/train/train_xgboost.py --root <PATH_TO_DATA> --year 2020 \
  --train-fraction 0.001 --n-estimators 50 --pixels-per-tile 1000

# Test HistGB (3 min)
python server/train/train_histgb.py --root <PATH_TO_DATA> --year 2020 \
  --train-fraction 0.001 --max-iter 50 --pixels-per-tile 1000

# Test RF (5 min)
python server/train/train_rf_baseline.py --root <PATH_TO_DATA> --year 2020 \
  --train-fraction 0.001 --rf-estimators 50 --pixels-per-tile 1000

# Test SVM (10 min)
python server/train/train_svm_baseline.py --root <PATH_TO_DATA> --year 2020 \
  --train-fraction 0.0005 --pixels-per-tile 500
```

### Production Training Workflow
Train all models with optimal settings:

```bash
# Create runs directory
mkdir -p runs

# Train XGBoost (40 min)
python server/train/train_xgboost.py --root <PATH_TO_DATA> --year 2020 \
  --train-fraction 0.01 --balance-pixels \
  --save-model ./runs/xgb_2020.joblib

# Train HistGB (30 min)
python server/train/train_histgb.py --root <PATH_TO_DATA> --year 2020 \
  --train-fraction 0.01 --balance-pixels \
  --save-model ./runs/hgb_2020.joblib

# Train RF (60 min)
python server/train/train_rf_baseline.py --root <PATH_TO_DATA> --year 2020 \
  --train-fraction 0.01 --balance-pixels \
  --save-model ./runs/rf_2020.joblib

# Skip SVM or use linear kernel for faster training (30 min)
python server/train/train_svm_baseline.py --root <PATH_TO_DATA> --year 2020 \
  --train-fraction 0.001 --balance-pixels --svm-kernel linear \
  --save-model ./runs/svm_linear_2020.joblib
```

---

## Integration with ml_utils

All training scripts use `ml_utils.py` for:
- `build_xy_from_tiles()`: Extract pixel features and labels from tiles
- `f1_iou()`: Compute F1 and IoU metrics

The pixel extraction process:
1. Loads each tile with its valid mask and wheat label
2. Samples up to `--pixels-per-tile` valid pixels
3. Optionally balances positive/negative pixels (with `--balance-pixels`)
4. Flattens temporal-band features into a single feature vector per pixel
5. Aggregates across all sampled tiles

---

## Notes

- All models use **stratified sampling** to preserve region proportions and wheat coverage distribution
- The 1% default (`--train-fraction 0.01`) is a good balance for most datasets
- Always use `--balance-pixels` for imbalanced datasets (where wheat is rare)
- Models are saved as `.joblib` files and can be loaded with `joblib.load(path)`
- For reproducibility, always set `--seed` to the same value
- The `--months` parameter should match your preprocessing configuration

---

## Getting Help

For detailed parameter information, use the `--help` flag:

```bash
python server/train/train_xgboost.py --help
python server/train/train_histgb.py --help
python server/train/train_rf_baseline.py --help
python server/train/train_svm_baseline.py --help
```
