# `train_histgb.py` – HistGradientBoosting Baseline

Scikit-learn histogram gradient boosting classifier for pixel-level wheat segmentation. Provides strong performance without external dependencies like XGBoost.

## Run Example
```powershell
python scripts\train_histgb.py --root "C:\Users\Administrator\Desktop\preprocessed_data" --year 2020 --train-fraction 0.01 --test-fraction 0.25 --pixels-per-tile 4096 --seed 42 --max-depth 8 --max-iter 400 --learning-rate 0.05 --save-model runs/hgb.joblib
```

## Flags
| Flag | Description |
|------|-------------|
| `--train-fraction` | Fraction of ALL tiles for training. |
| `--test-fraction`  | Fraction of remaining tiles for test. |
| `--pixels-per-tile`| Max valid pixels sampled per tile. |
| `--balance-pixels` | Balance positives/negatives per tile. |
| `--max-depth`      | Max tree depth (None for unlimited). |
| `--max-iter`       | Boosting iterations (trees). |
| `--learning-rate`  | Shrinkage factor. |
| `--l2-regularization` | L2 penalty. |
| `--save-model`     | Save trained model artifact. |

## Notes
- Uses `loss="log_loss"` for binary classification.
- Provides `predict_proba` in recent scikit-learn versions; fallback to `decision_function` if missing.

## Debug Tips
- Lower `--pixels-per-tile` to 512 or 256 for speed.
- Raise `--train-fraction` for better generalization once working.
