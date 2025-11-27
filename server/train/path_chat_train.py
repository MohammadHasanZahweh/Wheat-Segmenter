# patch_train.py

import joblib
from pathlib import Path
from typing import Dict, Any
import numpy as np

from server.server.config import (
    ModelType,
    SklearnType,
    ModelRun,
    PATCH_SPLIT_DATA_PATH,   # <-- assumed analogous to PIXEL_SPLIT_DATA_PATH
    PixelTrainRequest,       # <-- assumed analogous to PixelTrainRequest
    MODELS_PATH,
    META_PATH,
)

# Use your patch dataset implementation here.
# Replace these with whatever classes you actually expose in PatchDataset.py.
from server.dataset.PatchDataset import (
    PatchRangeNPYDataset,    # e.g. range-based patch dataset
    PatchFromKNPYDataset,    # e.g. "K" or held-out patch dataset
)

from server.model.sklearn_pixel_model import SklearnModel
from server.model.torch_pixel_model import TorchPixelPatchModel
from server.model.RNNmodel import RNNPixelPatchModel


# -------------------------------------------------------
# Train routine for PATCH model
# -------------------------------------------------------
def train_patch_model(req: PatchTrainRequest) -> Dict[str, Any]:
    """
    High-level training routine for patch-based models.

    Expected behavior (mirrors train_pixel_model):

        - Load train / val / test patch datasets
        - Build model (Sklearn / Torch / RNN)
        - Fit on patches via model.fit_patch(...)
        - Evaluate via model.val_patch_dataset(...)
        - Save model to MODELS_PATH
    """
    print("[INFO] Loading PATCH datasets...")

    root = PATCH_SPLIT_DATA_PATH / req.class_root

    # These dataset classes are assumed to have the same interface as your
    # pixel datasets:
    #   - .class_names
    #   - .mean
    #   - .std
    #   - __len__ and __getitem__ returning (X, Y) with
    #       X: (k, 9, 13, H, W)
    #       Y: (k, H, W)
    train_ds = PatchRangeNPYDataset(
        root,
        0,
        req.train_batches - 1,
        meta_files=META_PATH,
    )
    val_ds = PatchRangeNPYDataset(
        root,
        req.train_batches,
        req.train_batches + req.val_batches - 1,
        meta_files=META_PATH,
    )
    test_ds = PatchFromKNPYDataset(
        root,
        req.train_batches + req.val_batches,
        meta_files=META_PATH,
    )

    print(f"Train size (tiles): {len(train_ds)}")
    print(f"Val size (tiles):   {len(val_ds)}")
    print(f"Test size (tiles):  {len(test_ds)}")

    status: Dict[str, Any] = {"status": "completed"}

    # -----------------------
    # Build model
    # -----------------------
    if req.model_type == ModelType.SKLEARN:
        # Your sklearn model is assumed to implement:
        #   - fit_patch(dataset)
        #   - val_patch_dataset(dataset)
        model = SklearnModel(req.sub_model_type)
        save_path = MODELS_PATH / (req.run_save_name + ".sklearn.joblib")

    elif req.model_type == ModelType.TORCH:
        save_path = MODELS_PATH / (req.run_save_name + ".torch.joblib")
        model = TorchPixelPatchModel(
            train_ds.class_names,
            mean=train_ds.mean,
            std=train_ds.std,
        )

    elif req.model_type == ModelType.RNN:
        save_path = MODELS_PATH / (req.run_save_name + ".rnn.joblib")
        model = RNNPixelPatchModel(
            train_ds.class_names,
            mean=train_ds.mean,
            std=train_ds.std,
        )

    else:
        raise NotImplementedError(
            f"Model type {req.model_type} not supported for patch training."
        )

    # -----------------------
    # Fit on PATCHES
    # -----------------------
    print("[INFO] Training on PATCH dataset...")
    # All models are expected to expose a fit_patch(dataset, epochs=...)
    model.fit_patch(train_ds)

    # -----------------------
    # Validation
    # -----------------------
    # val_patch_dataset is expected to return:
    #   { "confusion_matrix": ..., "accuracy": ..., "F1_score": ... }
    print("[INFO] Running validation on PATCH dataset...")
    val_metrics = model.val_patch_dataset(val_ds)

    # Prefix metrics with "val_" so they match the pixel training status keys
    for k, v in val_metrics.items():
        status["val_" + k] = v

    if "val_accuracy" in status:
        print(f"[INFO] Validation accuracy = {status['val_accuracy']:.4f}")

    # -----------------------
    # Save
    # -----------------------
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    status["save_path"] = str(save_path)

    # -----------------------
    # Final test evaluation
    # -----------------------
    print("[INFO] Evaluating on TEST patch dataset...")
    test_metrics = model.val_patch_dataset(test_ds)
    for k, v in test_metrics.items():
        status["test_" + k] = v

    if "test_accuracy" in status:
        print(f"[INFO] Test accuracy = {status['test_accuracy']:.4f}")

    return status


if __name__ == "__main__":
    import sys
    import os

    # Allow running this file directly like pixel_train.py
    sys.path.append("..")
    os.chdir("../..")

    # Example call – adjust to your actual PatchTrainRequest fields
    from server.server.config import PatchTrainRequest, ModelType, SklearnType

    result = train_patch_model(
        PatchTrainRequest(
            run_save_name="patch_t1",
            model_type=ModelType.TORCH,
            sub_model_type=SklearnType.LR,  # unused for torch, but may be required by dataclass
            val_batches=1,
            train_batches=1,
            class_root="wheat",
        )
    )
    print(result)
