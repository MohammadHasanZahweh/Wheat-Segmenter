import os
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Dict
from pydantic import BaseModel


DATA_PATH   = Path(os.getenv("DATA_PATH", r"D:\image_lebanon\Lebanon\blbk_dataset\Requested_Tiffs_lcc"))
MODELS_PATH = Path(os.getenv("RUNS_PATH", r"../runs"))
PROCESS_DATA_PATH       = Path(r"../processed_data/")
PIXEL_SPLIT_DATA_PATH   = PROCESS_DATA_PATH/"split_processed_data"


# ---------------------------
# ENUM FOR MODEL TYPES
# ---------------------------
class ModelType(Enum):
    ONNX = "onnx_model"
    SKLEARN = "sklearn_pipeline"
    XGBOOST = "xgboost_model"
    TORCH = "torch_model"   # currently not implemented


# ---------------------------
# DATA CLASS FOR MODEL RUN
# ---------------------------
@dataclass
class ModelRun:
    model: Any
    model_type: ModelType
    save_path: Path
    metadata: Optional[Dict[str, Any]] = None


class SklearnType(Enum):
    KNN = "KNN"
    LR = "LR"
    METHOD3 = "M3"
    METHOD4 = "M4"   

class TrainRequest(BaseModel):
    ## TODO: add batch size
    ## TODO: add Torch types and XGBOOST
    run_save_name: str
    model_type: ModelType
    sub_model_type: SklearnType #| TorchType | XGBOOSTType
    val_batches: int
    train_batches: int
    class_root: str = "wheat"