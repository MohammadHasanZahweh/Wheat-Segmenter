from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import base64
from functools import partial
import threading
import time
from uuid import uuid4
from shapely.geometry import Polygon
from .config import MODELS_PATH, DATA_PATH,PATCH_SPLIT_DATA_PATH, RESULTS_DIR, LebanonInferenceRequest, TileDatasetConfig, TileTrainRequest, TrainingAlgorithm, YearInferenceRequest, TrainRequest
from server.train.sklearn_train import TrainConfig, train_sklearn_model
from server.inference.poly_tile_inference import run_on_multiple_tiles
from server.inference.inference_lebanon import run_on_lebanon_one_year
from server.model.torch_pixel_model import TorchPixelPatchModel
from server.model.sklearn_models import load_model as load_sklearn_model
import joblib

import logging 

logger = logging.getLogger()

app = FastAPI(title="Wheat Mapping API")


@app.get("/")
def root():
    return {"message": "Wheat Mapping API", "status": "running", "endpoints": ["/health", "/train", "/inference", "/inference-lebanon", "/models/list"]}


def _dataset_kwargs(cfg: TileDatasetConfig) -> dict[str, Any]:
    """Extract dataset kwargs from TileDatasetConfig."""
    # Use project_name to construct root path
    root_path = str(DATA_PATH.parent / "preprocessed_data" / cfg.project_name)
    
    kwargs = {
        "root_preprocessed": root_path,
        "year": str(cfg.year),
        "regions": cfg.regions,
        "month_order": tuple(cfg.months),
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
    
    # Get project name from dataset or use default
    project_name = req.dataset.project_name if req.dataset else "default"
    
    # If user provided full path with extension, use it as-is
    if req.output_path:
        output = str(Path(req.output_path))
        # Auto-add .sklearn.joblib if they only provided base name
        if not output.endswith('.sklearn.joblib') and not output.endswith('.joblib'):
            output = output + '.sklearn.joblib'
        elif output.endswith('.joblib') and not output.endswith('.sklearn.joblib'):
            output = output.replace('.joblib', '.sklearn.joblib')
        return output
    
    # Auto-generate path: runs/{project_name}/{job_name}.sklearn.joblib
    job_name = req.job_name
    # Remove any existing extensions user might have added
    for ext in ['.sklearn.joblib', '.joblib', '.sklearn']:
        if job_name.endswith(ext):
            job_name = job_name[:-len(ext)]
    
    default_name = f"{job_name}.sklearn.joblib"
    project_dir = MODELS_PATH / project_name
    return str((project_dir / default_name).resolve())


def _run_training_job(req: TileTrainRequest) -> dict[str, Any]:
    """Unified training job runner using consolidated sklearn_train module."""
    # Map API algorithm to TrainConfig model_type
    model_type_map = {
        TrainingAlgorithm.SVM: "svm",
        TrainingAlgorithm.RANDOM_FOREST: "random_forest",
        TrainingAlgorithm.HISTOGRAM_GB: "histgb",
        TrainingAlgorithm.XGBOOST: "xgboost",
    }
    
    # Use the actual preprocessed_data path
    root_path = PATCH_SPLIT_DATA_PATH
    
    # Build base config
    cfg = TrainConfig(
        model_type=model_type_map[req.algorithm],
        root=root_path,
        year=str(req.dataset.year),
        regions=req.dataset.regions,
        months=tuple(req.dataset.months),
        train_fraction=req.dataset.train_fraction,
        test_fraction=req.dataset.test_fraction,
        pixels_per_tile=req.dataset.pixels_per_tile,
        balance_pixels=req.dataset.balance_pixels,
        seed=req.dataset.seed,
        use_meta_stats=bool(req.dataset.meta_dir),
        meta_dir=req.dataset.meta_dir or "./meta",
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


def run_inference_job(job_id: str, data_path: Path, year: int, aois: list, model, output_path: Path, polygon) -> None:
    try:
        print(f"\n{'='*60}", flush=True)
        print(f"[API INFERENCE] Starting job {job_id}", flush=True)
        print(f"  Data: {data_path}", flush=True)
        print(f"  Year: {year}, AOIs: {aois}", flush=True)
        print(f"  Output: {output_path}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        run_on_multiple_tiles(
            base_path=str(data_path),
            year=year,
            aois=aois,
            model=model,
            out_path=output_path,
            polygons=polygon,
            patch_size=256,
            stride=256
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = str(output_path)
        
        print(f"\n{'='*60}", flush=True)
        print(f"[API INFERENCE] Job {job_id} completed!", flush=True)
        print(f"  Output saved to: {output_path}", flush=True)
        print(f"{'='*60}\n", flush=True)
    except Exception as exc:
        print(f"\n[API INFERENCE] Job {job_id} FAILED: {exc}\n", flush=True)
        jobs[job_id].update({"status": "failed", "error": str(exc)})

def run_inference_job_lebanon(job_id: str, data_path: Path, year: int, model, output_path: Path, polygons) -> None:
    try:
        print(f"\n{'='*60}", flush=True)
        print(f"[API INFERENCE] Starting job {job_id}", flush=True)
        print(f"  Data: {data_path}", flush=True)
        print(f"  Year: {year}", flush=True)
        print(f"  Output: {output_path}", flush=True)
        print(f"{'='*60}\n", flush=True)
        

        run_on_lebanon_one_year(
            base_path=data_path,
            year=year,
            polygons=polygons,
            process_fn=partial(model.predict_patch, normalize = True),
            out_path=output_path,
            patch_size=256,
            stride=256
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = str(output_path)
        
        print(f"\n{'='*60}", flush=True)
        print(f"[API INFERENCE] Job {job_id} completed!", flush=True)
        print(f"  Output saved to: {output_path}", flush=True)
        print(f"{'='*60}\n", flush=True)
    except Exception as exc:
        print(f"\n[API INFERENCE] Job {job_id} FAILED: {exc}\n", flush=True)
        jobs[job_id].update({"status": "failed", "error": str(exc)})

@app.post("/train")
def start_train(req: TileTrainRequest):
    try:
        job_id = f"job_{uuid4().hex}"
        jobs[job_id] = {
            "status": "running",
            "job_name": req.job_name,
            "algorithm": req.algorithm.value,
            "submitted_at": time.time(),
        }

        payload = req.model_dump()  
        
        thread = threading.Thread(target=train_job, args=(job_id, payload), daemon=True)
        thread.start()
        return {"job_id": job_id, "status": "running"}
    except Exception as e:
        print(f"[ERROR] Failed to start training: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


@app.get("/train/status")
def train_status(id: str):
    return jobs.get(id, {"status": "unknown"})

@app.post("/inference")
def start_inference(req: YearInferenceRequest):
    job_id = f"job_{uuid4().hex}"

    if not (req.save_name.endswith(".tiff") or req.save_name.endswith(".tif")):
        req.save_name += ".tiff"

    # Auto-detect model type based on extension pattern
    model_name = req.model_name
    model_path = None
    model_type = None
    
    # Determine model type by checking for .sklearn or .torch in the name
    if '.sklearn' in model_name:
        # Sklearn model
        model_type = "sklearn"
        # Ensure proper extension
        if not model_name.endswith('.sklearn.joblib'):
            if model_name.endswith('.sklearn'):
                model_name = model_name + '.joblib'
            elif model_name.endswith('.joblib'):
                model_name = model_name.replace('.joblib', '.sklearn.joblib')
            else:
                model_name = model_name + '.sklearn.joblib'
        model_path = MODELS_PATH / req.project_name / model_name
        
    elif '.torch' in model_name:
        # Torch model
        model_type = "torch"
        # Ensure proper extension
        if not model_name.endswith('.torch.joblib'):
            if model_name.endswith('.torch'):
                model_name = model_name + '.joblib'
            elif model_name.endswith('.joblib'):
                model_name = model_name.replace('.joblib', '.torch.joblib')
            else:
                model_name = model_name + '.torch.joblib'
        model_path = MODELS_PATH / req.project_name / model_name
        
    else:
        # No extension provided - try both
        sklearn_path = MODELS_PATH / req.project_name / (model_name + '.sklearn.joblib')
        torch_path = MODELS_PATH / req.project_name / (model_name + '.torch.joblib')
        
        if sklearn_path.exists():
            model_path = sklearn_path
            model_type = "sklearn"
            model_name = model_name + '.sklearn.joblib'
        elif torch_path.exists():
            model_path = torch_path
            model_type = "torch"
            model_name = model_name + '.torch.joblib'
        else:
            return {"job_id": job_id, "status": "failed", 
                    "reason": f"Model not found. Tried: {sklearn_path.name} and {torch_path.name}"}
    
    if model_path is None or not model_path.exists():
        return {"job_id": job_id, "status": "failed", "reason": f"Model not found: {model_path}"}
    
    try:
        print(f"[INFERENCE] Loading {model_type} model from {model_path}")
        
        if model_type == "sklearn":
            # Load sklearn model with meta_dir for normalization
            meta_dir = str(DATA_PATH / "meta") if (DATA_PATH / "meta").exists() else None
            model = load_sklearn_model(str(model_path), meta_dir=meta_dir)
            print(f"[INFERENCE] sklearn model loaded successfully: {model.model_type}")
        else:
            # Load torch model
            model = TorchPixelPatchModel.load(model_path)
            print(f"[INFERENCE] torch model loaded successfully")
            
    except Exception as e:
        return {"job_id": job_id, "status": "failed", "reason": f"Unable to load model: {str(e)}"}
    
    jobs[job_id] = {
        "status": "running",
        "job_name": req.project_name + "_" + req.save_name,
        "model_name": model_name,
        "model_type": model_type,
        "submitted_at": time.time(),
    }
    
    data_path = DATA_PATH / req.region_name / "download"
    output_path = RESULTS_DIR / req.project_name / req.save_name

    coords = req.geometry.coordinates[0]  # outer ring
    poly = Polygon(coords)
    
    thread = threading.Thread(
        target=run_inference_job, 
        args=(job_id, data_path, req.year, [0, 1, 2, 3, 4], model, output_path, poly), 
        daemon=True
    )
    thread.start()
    return {"job_id": job_id, "status": "running"}


@app.post("/inference-lebanon")
def start_inference_lebanon(req: LebanonInferenceRequest):
    job_id = f"job_{uuid4().hex}"

    print(req)

    if not (req.save_name.endswith(".tiff") or req.save_name.endswith(".tif")):
        req.save_name += ".tiff"

    # Auto-detect model type based on extension pattern
    model_name = req.model_name
    model_path = None
    model_type = None
    
    # Determine model type by checking for .sklearn or .torch in the name
    if '.sklearn' in model_name:
        # Sklearn model
        model_type = "sklearn"
        # Ensure proper extension
        if not model_name.endswith('.sklearn.joblib'):
            if model_name.endswith('.sklearn'):
                model_name = model_name + '.joblib'
            elif model_name.endswith('.joblib'):
                model_name = model_name.replace('.joblib', '.sklearn.joblib')
            else:
                model_name = model_name + '.sklearn.joblib'
        model_path = MODELS_PATH / req.project_name / model_name
        
    elif '.torch' in model_name:
        # Torch model
        model_type = "torch"
        # Ensure proper extension
        if not model_name.endswith('.torch.joblib'):
            if model_name.endswith('.torch'):
                model_name = model_name + '.joblib'
            elif model_name.endswith('.joblib'):
                model_name = model_name.replace('.joblib', '.torch.joblib')
            else:
                model_name = model_name + '.torch.joblib'
        model_path = MODELS_PATH / req.project_name / model_name
        
    else:
        # No extension provided - try both
        sklearn_path = MODELS_PATH / req.project_name / (model_name + '.sklearn.joblib')
        torch_path = MODELS_PATH / req.project_name / (model_name + '.torch.joblib')
        
        if sklearn_path.exists():
            model_path = sklearn_path
            model_type = "sklearn"
            model_name = model_name + '.sklearn.joblib'
        elif torch_path.exists():
            model_path = torch_path
            model_type = "torch"
            model_name = model_name + '.torch.joblib'
        else:
            return {"job_id": job_id, "status": "failed", 
                    "reason": f"Model not found. Tried: {sklearn_path.name} and {torch_path.name}"}
    
    if model_path is None or not model_path.exists():
        return {"job_id": job_id, "status": "failed", "reason": f"Model not found: {model_path}"}
    
    try:
        print(f"[INFERENCE] Loading {model_type} model from {model_path}")
        
        if model_type == "sklearn":
            # Load sklearn model with meta_dir for normalization
            meta_dir = str(DATA_PATH / "meta") if (DATA_PATH / "meta").exists() else None
            model = load_sklearn_model(str(model_path), meta_dir=meta_dir)
            print(f"[INFERENCE] sklearn model loaded successfully: {model.model_type}")
        else:
            # Load torch model
            model = TorchPixelPatchModel.load(model_path)
            print(f"[INFERENCE] torch model loaded successfully")
            
    except Exception as e:
        return {"job_id": job_id, "status": "failed", "reason": f"Unable to load model: {str(e)}"}
    
    jobs[job_id] = {
        "status": "running",
        "job_name": req.project_name + "_" + req.save_name,
        "model_name": model_name,
        "model_type": model_type,
        "submitted_at": time.time(),
    }
    
    data_path = DATA_PATH / "Lebanon/merge_data"
    output_path = RESULTS_DIR / req.project_name / req.save_name
    coords = req.geometry.coordinates[0]  # outer ring
    poly = Polygon(coords)

    thread = threading.Thread(
        target=run_inference_job_lebanon, 
        args=(job_id, data_path, req.year,  model, output_path,poly), 
        daemon=True
    )
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
def get_results(project: str, run: str):
    if not run.endswith(".tiff"):
        run += ".tiff"
    
    result_path = RESULTS_DIR / project / run
    if result_path.exists():
        with open(result_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return JSONResponse(content={"image_base64": encoded, "status": "OK"})
    else:
        return {"status": "failed", "error": f"Result not found: {project}/{run}"}


@app.get("/models/list")
def list_models():
    """List all available trained models."""
    models = []
    if MODELS_PATH.exists():
        for model_file in MODELS_PATH.glob("**/*.joblib"):
            try:
                # Load model metadata
                loaded = joblib.load(model_file)
                
                model_info = {
                    "name": model_file.name,
                    "path": str(model_file.relative_to(MODELS_PATH)),
                    "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                }
                
                # Extract metadata if available
                if isinstance(loaded, dict):
                    model_info["model_type"] = loaded.get("model_type", "unknown")
                    model_info["year"] = loaded.get("year", "unknown")
                    model_info["has_meta_stats"] = loaded.get("has_meta_stats", False)
                    if "train_config" in loaded:
                        model_info["train_config"] = loaded["train_config"]
                else:
                    model_info["model_type"] = "legacy_format"
                
                models.append(model_info)
            except Exception as e:
                models.append({
                    "name": model_file.name,
                    "path": str(model_file.relative_to(MODELS_PATH)),
                    "error": f"Failed to load: {str(e)}"
                })
    
    return {"models": models, "count": len(models)}


@app.get("/models/info")
def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    if not model_name.endswith(".joblib"):
        model_name += ".joblib"
    
    model_path = MODELS_PATH / model_name
    if not model_path.exists():
        return {"status": "failed", "error": f"Model not found: {model_name}"}
    
    try:
        loaded = joblib.load(model_path)
        
        info = {
            "name": model_name,
            "path": str(model_path),
            "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        }
        
        if isinstance(loaded, dict):
            info["model_type"] = loaded.get("model_type", "unknown")
            info["year"] = loaded.get("year", "unknown")
            info["months"] = loaded.get("months", [])
            info["has_meta_stats"] = loaded.get("has_meta_stats", False)
            info["meta_dir"] = loaded.get("meta_dir", None)
            info["train_config"] = loaded.get("train_config", {})
            
            # Get model class info
            if "model" in loaded:
                model_obj = loaded["model"]
                info["model_class"] = model_obj.__class__.__name__
        else:
            info["model_type"] = "legacy_format"
            info["model_class"] = loaded.__class__.__name__
        
        return {"status": "success", "model": info}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    

