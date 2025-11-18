# `train_xgboost.py` – XGBoost Wheat Segmentation Baseline

Trains a pixel-level wheat vs. non-wheat classifier using XGBoost on a stratified subset of tiles and evaluates on a separate stratified test subset.

## Data Assumption
`--root` points to a folder containing:
```
<ROOT>/data/<YEAR>/<REGION>/<MONTH>/<TILE_ID>.tif
<ROOT>/label/<YEAR>/<REGION>/<TILE_ID>.tif    # 2 bands: valid, wheat
```

## Features
- Concatenated bands across selected months → per-pixel feature vector of length T*B.
- Only valid pixels (valid_mask > 0.5) sampled.
- Optional class balancing per tile.

## Metrics
Binary F1 and IoU at pixel level (threshold 0.5).

## Install (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Example Run
```powershell
python scripts\train_xgboost.py --root "C:\Users\Administrator\Desktop\preprocessed_data" --year 2020 --train-fraction 0.01 --test-fraction 0.25 --pixels-per-tile 4096 --seed 42 --n-estimators 400 --max-depth 8 --learning-rate 0.05 --save-model runs/xgb.joblib
```

## Key Flags
| Flag | Meaning |
|------|---------|
| `--train-fraction` | Fraction of ALL tiles sampled for training. |
| `--test-fraction`  | Fraction of remaining tiles for test. |
| `--pixels-per-tile`| Max valid pixels sampled per tile. |
| `--balance-pixels` | Attempt equal pos/neg pixels per tile. |
| `--n-estimators`   | Number of boosting rounds. |
| `--learning-rate`  | Shrinkage factor. |
| `--max-depth`      | Tree depth (controls complexity). |
| `--subsample`      | Row subsampling for each tree. |
| `--colsample-bytree` | Feature subsampling per tree. |
| `--save-model`     | Path to save `.joblib` artifact. |

## Testing Quickness
Reduce `--pixels-per-tile 512` for faster debugging.

## Extend
Tune `--max-depth`, `--learning-rate`, or add early stopping with a validation split (modify script: pass `eval_set`).
