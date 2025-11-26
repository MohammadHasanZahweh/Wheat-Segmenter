from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import base64

import threading
import time
from uuid import uuid4

from .config import MODELS_PATH, DATA_PATH, RESULTS_DIR, TileDatasetConfig, TileTrainRequest, TrainingAlgorithm, YearInferenceRequest, TrainRequest
from server.train.sklearn_train import TrainConfig, train_sklearn_model
from server.inference.tile_inference import run_on_multiple_tiles
from server.model.torch_pixel_model import TorchPixelPatchModel

import logging 

logger = logging.getLogger()

app = FastAPI(title="Wheat Mapping API")


def _dataset_kwargs(cfg: TileDatasetConfig) -> dict[str, Any]:
    """Extract dataset kwargs from TileDatasetConfig."""
    kwargs = {
        "project_name": cfg.root,
        "year": cfg.year,
        "regions": cfg.regions,
        "months": tuple(cfg.months),
        "train_fraction": cfg.train_fraction,
        "test_fraction": cfg.test_fraction,
        "pixels_per_tile": cfg.pixels_per_tile,
        "balance_pixels": cfg.balance_pixels,
        "seed": cfg.seed,
        "normalize": cfg.normalize,
    }
    # Add meta stats support (TrainConfig expects use_meta_stats and meta_dir, not band_stats)
    kwargs["meta_dir"] = cfg.meta_dir

    return kwargs


def _resolve_save_path(req: TileTrainRequest) -> str | None:
    if not req.save_model:
        return None
    if req.output_path:
        return str(Path(req.output_path))
    default_name = f"{req.job_name}_{req.algorithm.value}.joblib"
    return str((MODELS_PATH / default_name).resolve())


def _run_training_job(req: TileTrainRequest) -> dict[str, Any]:
    """Unified training job runner using consolidated sklearn_train module."""
    # Map API algorithm to TrainConfig model_type
    model_type_map = {
        TrainingAlgorithm.SVM: "svm",
        TrainingAlgorithm.RANDOM_FOREST: "random_forest",
        TrainingAlgorithm.HISTOGRAM_GB: "hist_gradient_boosting",
        TrainingAlgorithm.XGBOOST: "xgboost",
    }
    
    # Build base config
    cfg = TrainConfig(
        model_type=model_type_map[req.algorithm],
        **_dataset_kwargs(req.dataset),
        save_model=_resolve_save_path(req),
    )
    
    # Override hyperparameters from request
    if req.model_params:
        for key, value in req.model_params.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    
    # Run training
    result = train_sklearn_model(cfg)
    result["algorithm"] = req.algorithm.value
    result["job_name"] = req.job_name
    return result


jobs: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def train_job(job_id: str, payload: dict[str, Any]) -> None:
    try:
        print(f"\n{'='*60}", flush=True)
        print(f"[API TRAINING] Starting job {job_id}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        req = TileTrainRequest(**payload)
        result = _run_training_job(req)
        jobs[job_id].update(result)
        jobs[job_id]["status"] = "completed"
        
        print(f"\n{'='*60}", flush=True)
        print(f"[API TRAINING] Job {job_id} completed!", flush=True)
        print(f"  F1: {result.get('f1', 0):.4f}, IoU: {result.get('iou', 0):.4f}", flush=True)
        print(f"{'='*60}\n", flush=True)
    except Exception as exc:  # pragma: no cover - surfaced via API
        print(f"\n[API TRAINING] Job {job_id} FAILED: {exc}\n", flush=True)
        jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.post("/train")
def start_train(req: TileTrainRequest):
    job_id = f"job_{uuid4().hex}"
    jobs[job_id] = {
        "status": "running",
        "job_name": req.job_name,
        "algorithm": req.algorithm.value,
        "submitted_at": time.time(),
    }
    # Pydantic v1 uses dict(), v2 uses model_dump()
    payload = req.model_dump()
    thread = threading.Thread(target=train_job, args=(job_id, payload), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "running"}


@app.get("/train/status")
def train_status(id: str):
    return jobs.get(id, {"status": "unknown"})

@app.post("/inference")
def start_train(req: YearInferenceRequest):
    job_id = f"job_{uuid4().hex}"

    if not req.model_name.endswith(".joblib"):
        req.model_name += ".joblib"

    if not (req.save_name.endswith(".tiff") or req.save_name.endswith(".tif")):
        req.save_name += ".tiff"

    try:
        print(MODELS_PATH/req.project_name/req.model_name)
        model = TorchPixelPatchModel.load(MODELS_PATH/req.project_name/req.model_name)

    except:
         return {"job_id": job_id, "status": "failed", "reason":"Unable to load model"}
    jobs[job_id] = {
        "status": "running",
        "job_name": req.project_name + "_" + req.save_name,
        "submitted_at": time.time(),
    }
    # payload = req.model_dump()
    thread = threading.Thread(target=run_on_multiple_tiles, args=(DATA_PATH/req.region_name/ "download" ,req.year,[0,1,2,3,4], model, RESULTS_DIR/req.project_name/req.save_name), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "running"}

@app.post("/pixel-train")
def start_train(req: TrainRequest):
    job_id = f"job_{int(time.time())}"
    jobs[job_id] = {"status": "running"}
    t = threading.Thread(target=train_job, args=(job_id, req.config_name), daemon=True)
    t.start()
    return {"job_id": job_id}

@app.get("/results")
def train_status(project: str, run:str):
    if not run.endswith(".tiff"):
        run += ".tiff"
    if (RESULTS_DIR/project/run).exists():

        with open(RESULTS_DIR/project/run, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")

        return JSONResponse(content={"image_base64": encoded, "status":"OK"})
    else:
        return {"status": f"failed to find image {project}/{run}"}
    