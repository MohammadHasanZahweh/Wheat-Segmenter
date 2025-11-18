# train.py
import joblib
from pathlib import Path
from typing import Dict, Any
import numpy as np

from server.config import ModelType, SklearnType, ModelRun, PIXEL_SPLIT_DATA_PATH, TrainRequest, MODELS_PATH
from dataset.PixelDataset import PixelRangeNPYDataset, PixelFromKNPYDataset

from model import keras_pixel_model


# -------------------------------------------------------
# Train routine
# -------------------------------------------------------
def train_pixel_model(req:TrainRequest) -> Dict[str, Any]:
    print("[INFO] Loading datasets...")

    root = PIXEL_SPLIT_DATA_PATH / req.class_root

    import os
    print(root)
    print(root.exists())
    print(os.listdir(root))
    print(req)
    

    train_ds = PixelRangeNPYDataset(root, 0, req.train_batches - 1)
    val_ds   = PixelRangeNPYDataset(root, req.train_batches, req.train_batches + req.val_batches - 1)
    test_ds  = PixelFromKNPYDataset(root, req.train_batches + req.val_batches )

    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    print(f"Test size:  {len(test_ds)}")
    
    # flatten numpy arrays for sklearn
    X_train = np.concatenate([x for x, _ in train_ds])
    X_train = X_train.reshape((X_train.shape[0],-1))
    y_train = np.concatenate([y * np.ones(((x.shape[0]))) for x, y in train_ds])

    X_val =  np.concatenate([x for x, _ in val_ds])
    X_val =  X_val.reshape((X_val.shape[0],-1))
    y_val = np.concatenate([y * np.ones(((x.shape[0]))) for x, y in val_ds])

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)

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
    save_path = MODELS_PATH / (req.run_save_name + ".joblib")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)

    print(save_path.absolute())

    conf = np.zeros(shape=(y_train.astype(np.int16).max()+1,y_train.astype(np.int16).max()+1))

    print("[TEST] Training...")
    for X,y in test_ds:
        yp = model.predict(X.reshape((X.shape[0],-1))).astype(np.int16)
        # for i in range(y.min(), y.max()+1):
        for j in range(yp.min(), yp.max()+1):
            conf[y,j] += ((yp==j)).sum()
    
    print(conf)

    return {
        "status": "completed",
        "val_acc": float(val_acc),
        "save_path": str(save_path),
        "confusion_matrix": conf.tolist(),
        "test_F1":2*conf[1,1]/(2*conf[1,1] + conf[1,0] + conf[0,1]),
        "test_accuracy": (conf[1,1] +  conf[0,0])/conf.sum()
    }

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    import os
    os.chdir("../server")
    train_pixel_model(TrainRequest(
        run_save_name="t1",
        model_type=ModelType.SKLEARN,
        sub_model_type=SklearnType.LogisticRegression,
        val_batches=1,
        train_batches=1,
        class_root = "wheat"
        ))