from config import MODELS_PATH, ModelRun, ModelType
import joblib
import xgboost as xgb
import onnx
import torch
from pathlib import Path

import numpy as np
from PIL import Image
import base64
import io

# ---------------------------
# SAVE MODEL FUNCTION
# ---------------------------
def save_model(run: ModelRun) -> None:
    """Save the model according to its type."""
    path = Path(run.save_path)

    if run.model_type == ModelType.ONNX:
        onnx.save(run.model, path.with_suffix(".onnx"))

    elif run.model_type == ModelType.SKLEARN:
        joblib.dump(run.model, path.with_suffix(".joblib"))

    elif run.model_type == ModelType.XGBOOST:
        run.model.save_model(path.with_suffix(".json"))  # recommended
        # run.model.save_model(path.with_suffix(".model"))  # if you want binary

    elif run.model_type == ModelType.TORCH:
        raise NotImplementedError("Torch model saving not implemented yet.")

    else:
        raise ValueError("Unknown model type")

    # Save metadata if provided
    if run.metadata:
        joblib.dump(run.metadata, path.with_suffix(".meta.joblib"))

    print(f"Model saved successfully to {path}")


# ---------------------------
# LOAD MODEL FUNCTION
# ---------------------------
def load_model(path: Path, model_type: ModelType):
    """Load a model based on type and path."""
    path = Path(path)

    if model_type == ModelType.ONNX:
        return onnx.load(path.with_suffix(".onnx"))

    elif model_type == ModelType.SKLEARN:
        return joblib.load(path.with_suffix(".joblib"))

    elif model_type == ModelType.XGBOOST:
        model = xgb.Booster()
        model.load_model(path.with_suffix(".json"))
        return model

    elif model_type == ModelType.TORCH:
        raise NotImplementedError("Torch model loading not implemented yet.")

    else:
        raise ValueError("Unknown model type")
    

def np_to_png_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")