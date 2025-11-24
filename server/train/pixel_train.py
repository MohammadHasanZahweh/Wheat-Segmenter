# train.py
import joblib
from pathlib import Path
from typing import Dict, Any
import numpy as np

from server.server.config import ModelType, SklearnType, ModelRun, PIXEL_SPLIT_DATA_PATH, PixelTrainRequest, MODELS_PATH
from server.dataset.PixelDataset import PixelRangeNPYDataset, PixelFromKNPYDataset
from server.model.sklearn_pixel_model import SklearnModel
from server.model.torch_pixel_model import TorchPixelPatchModel


# -------------------------------------------------------
# Train routine
# -------------------------------------------------------
def train_pixel_model(req: PixelTrainRequest) -> Dict[str, Any]:
    print("[INFO] Loading datasets...")

    root = PIXEL_SPLIT_DATA_PATH / req.class_root
    

    train_ds = PixelRangeNPYDataset(root, 0, req.train_batches - 1)
    val_ds   = PixelRangeNPYDataset(root, req.train_batches, req.train_batches + req.val_batches - 1)
    test_ds  = PixelFromKNPYDataset(root, req.train_batches + req.val_batches )

    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    print(f"Test size:  {len(test_ds)}")

    status = {
        "status": "completed",
    }

    # -----------------------
    # Build model
    # -----------------------
    if req.model_type == ModelType.SKLEARN:
        model = SklearnModel(req.sub_model_type)
    
    elif req.model_type == ModelType.TORCH:
        model = TorchPixelPatchModel(train_ds.class_names, mean=train_ds.mean, std=train_ds.std)
    
    else:
        raise NotImplementedError(f"Model type {req.model_type} not supported yet.")

    # -----------------------
    # Fit
    # -----------------------
    print("[INFO] Training...")
    model.fit_pixel(train_ds)

    status.update(model.val_pixel_dataset(val_ds,"val_"))
    print(f"[INFO] Validation accuracy = {status["val_accuracy"]:.4f}")

    # -----------------------
    # Save
    # -----------------------
    save_path = MODELS_PATH / (req.run_save_name + ".joblib")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    # joblib.dump(model, save_path)
    status["save_path"] = str(save_path)



    status.update(model.val_pixel_dataset(test_ds, "test_"))


    return status

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    import os
    os.chdir("../server")
    a = train_pixel_model(PixelTrainRequest(
        run_save_name="t1",
        model_type=ModelType.SKLEARN,
        sub_model_type=SklearnType.LR,
        val_batches=1,
        train_batches=1,
        class_root = "wheat",
        ))
    print(a)
