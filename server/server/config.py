import os
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field


DATA_PATH   = Path(os.getenv("DATA_DIR", r"./data"))
META_PATH   = DATA_PATH / "meta/wheat"
PROCESS_DATA_PATH       = DATA_PATH/"processed_data"
PIXEL_SPLIT_DATA_PATH   = PROCESS_DATA_PATH/"split_processed_data"

MODELS_PATH = Path(os.getenv("RUNS_DIR", r"./runs"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", r"./results"))


# ---------------------------
# ENUM FOR MODEL TYPES
# ---------------------------
class ModelType(Enum):
    ONNX    = "onnx_model"
    SKLEARN = "sklearn_pipeline"
    XGBOOST = "xgboost_model"
    TORCH   = "torch_model"   # currently not implemented


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


class PixelTrainRequest(BaseModel):
    """
    Legacy pixel-level training request used by pixel_sklearn_train.
    """

    run_save_name: str
    model_type: ModelType
    sub_model_type: SklearnType  # | TorchType | XGBOOSTType
    val_batches: int
    train_batches: int
    class_root: str = "wheat"


class TrainingAlgorithm(str, Enum):
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    HISTOGRAM_GB = "hist_gradient_boosting"
    XGBOOST = "xgboost"
    LR = "LogisticRegression"



class TileDatasetConfig(BaseModel):
    project_name: str
    year: int
    regions: Optional[List[str]] = None
    months: List[int] = Field(
        default_factory=lambda: [11, 12, 1, 2, 3, 4, 5, 6, 7]
    )
    train_fraction: float = 0.01
    test_fraction: float = 0.25
    pixels_per_tile: int = 4096
    balance_pixels: bool = False
    seed: int = 42
    normalize: bool = True
    meta_dir: Optional[str] = None


class TileTrainRequest(BaseModel):
    """
    Request payload for API-triggered tile-based training jobs.
    """

    job_name: str
    algorithm: TrainingAlgorithm
    dataset: TileDatasetConfig | None
    save_model: bool = True
    output_path: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None

WHEAT_DATASET = TileDatasetConfig(
    project_name = "wheat",
    year=2020,
    regions = None,
    months=[11, 12, 1, 2, 3, 4, 5, 6, 7],
    train_fraction = 0.01,
    test_fraction = 0.25,
    balance_pixels = False,
    seed = 42,
    normalize=True,
)

class TrainRequest(BaseModel):
    """
    Request payload for API-triggered tile-based training jobs.
    """

    job_name: str
    algorithm: TrainingAlgorithm
    dataset: TileDatasetConfig = WHEAT_DATASET
    model_name: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None

class YearInferenceRequest(BaseModel):
    """
    Request payload for API-triggered year jobs.
    """
    region_name:str
    project_name: str
    model_name : str
    year:int
    save_name:str
    # dataset: TileDatasetConfig

