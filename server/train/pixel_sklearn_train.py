# train.py
import joblib
from pathlib import Path
from typing import Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from server.config import ModelType, SklearnType, ModelRun, PIXEL_SPLIT_DATA_PATH, TrainRequest
from dataset.PixelDataset import PixelRangeNPYDataset, PixelFromKNPYDataset


# -------------------------------------------------------
# Build sklearn sub-model
# -------------------------------------------------------
def build_sklearn_model(sub: SklearnType):
    if sub == SklearnType.KNN:
        return KNeighborsClassifier(n_neighbors=3)
    if sub == SklearnType.LR:
        return LogisticRegression(max_iter=500)

    raise NotImplementedError(f"Sub-model type {sub} is not implemented.")


# -------------------------------------------------------
# Train routine
# -------------------------------------------------------
def train_pixel_model(req:TrainRequest) -> Dict[str, Any]:
    print("[INFO] Loading datasets...")

    root = PIXEL_SPLIT_DATA_PATH / req.class_root

    train_ds = PixelRangeNPYDataset(root, 0, req.train_batches)
    val_ds   = PixelRangeNPYDataset(root, req.train_batches, req.train_batches + req.val_batches)
    test_ds  = PixelFromKNPYDataset(root, req.train_batches + req.val_batches)

    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    print(f"Test size:  {len(test_ds)}")

    # flatten numpy arrays for sklearn
    X_train = [x.flatten() for x, _ in train_ds]
    y_train = [y for _, y in train_ds]

    X_val = [x.flatten() for x, _ in val_ds]
    y_val = [y for _, y in val_ds]

    # -----------------------
    # Build model
    # -----------------------
    if req.model_type == ModelType.SKLEARN:
        model = build_sklearn_model(req.sub_model_type)
    else:
        raise NotImplementedError(f"Model type {req.model_type} not supported yet.")

    # -----------------------
    # Fit
    # -----------------------
    print("[INFO] Training...")
    model.fit(X_train, y_train)

    val_acc = model.score(X_val, y_val)
    print(f"[INFO] Validation accuracy = {val_acc:.4f}")

    # -----------------------
    # Save
    # -----------------------
    save_path = Path("runs") / (req.run_save_name + ".joblib")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)

    return {
        "status": "completed",
        "val_acc": float(val_acc),
        "save_path": str(save_path)
    }

if __name__ == "__main__":

    train_model(TrainRequest(
        run_save_name="t1",
        model_type=ModelType.SKLEARN,
        sub_model_type=SklearnType.LogisticRegression,
        val_batches=1,
        train_batches=1,
        class_root = "wheat"
        ))