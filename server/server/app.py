from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict

from fastapi import FastAPI

import threading
import time
from uuid import uuid4

from .config import MODELS_PATH, TileDatasetConfig, TileTrainRequest, TrainingAlgorithm, YearInferenceRequest
from server.train.sklearn_train import TrainConfig, train_sklearn_model
from server.inference.tile_inference import run_on_multiple_tiles
app = FastAPI(title="Wheat Mapping API")


def _dataset_kwargs(cfg: TileDatasetConfig) -> dict[str, Any]:
    """Extract dataset kwargs from TileDatasetConfig."""
    kwargs = {
        "root": cfg.root,
        "year": cfg.year,
        "regions": cfg.regions,
        "months": tuple(cfg.months),
        "train_fraction": cfg.train_fraction,
        "test_fraction": cfg.test_fraction,
        "pixels_per_tile": cfg.pixels_per_tile,
        "balance_pixels": cfg.balance_pixels,
        "seed": cfg.seed,
    }
    # Add meta stats support (TrainConfig expects use_meta_stats and meta_dir, not band_stats)
    if cfg.use_meta_stats:
        kwargs["use_meta_stats"] = True
        kwargs["meta_dir"] = cfg.meta_dir or "./meta"
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
    payload = req.dict() if hasattr(req, 'dict') else req.model_dump()
    thread = threading.Thread(target=train_job, args=(job_id, payload), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "running"}


@app.get("/train/status")
def train_status(id: str):
    return jobs.get(id, {"status": "unknown"})

@app.post("/inference")
def start_train(req: YearInferenceRequest):
    job_id = f"job_{uuid4().hex}"
    
    try:
        from server.model.torch_pixel_model import TorchPixelPatchModel
        model = TorchPixelPatchModel.load(req.model_path)
    except:
         return {"job_id": job_id, "status": "failed", "reason":"Unable to load model"}
    jobs[job_id] = {
        "status": "running",
        "job_name": req.job_name,
        "submitted_at": time.time(),
    }
    payload = req.model_dump()
    thread = threading.Thread(target=run_on_multiple_tiles, args=(req.year,[0,1,2,3,4], model, req.save_path), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "running"}


# @app.post("/train")
# def start_train(req: TileTrainRequest):
#     job_id = f"job_{uuid4().hex}"
#     jobs[job_id] = {
#         "status": "running",
#         "job_name": req.job_name,
#         "algorithm": req.algorithm.value,
#         "submitted_at": time.time(),
#     }
#     payload = req.model_dump()
#     thread = threading.Thread(target=train_job, args=(job_id, payload), daemon=True)
#     thread.start()
#     return {"job_id": job_id, "status": "running"}

# jobs: dict[str, dict] = {}


# def train_job(job_id: str, config_name: str):
#     try:
#         cfg = load_yaml_config(Path("configs") / f"{config_name}.yaml")
#         res = train_from_config(cfg, data_root=cfg.get("data", {}).get("root", "data/processed/patches"))
#         jobs[job_id].update({"status": "completed", **res})
#     except Exception as e:
#         jobs[job_id].update({"status": "failed", "error": str(e)})


# @app.post("/train")
# def start_train(req: TrainRequest):
#     job_id = f"job_{int(time.time())}"
#     jobs[job_id] = {"status": "running"}
#     t = threading.Thread(target=train_job, args=(job_id, req.config_name), daemon=True)
#     t.start()
#     return {"job_id": job_id}


# @app.get("/train/status")
# def train_status(id: str):
#     return jobs.get(id, {"status": "unknown"})

# @app.get("/config/{name}")
# def get_config(name: str) -> Dict:
#     path = Path("configs") / f"{name}.yaml"
#     if not path.exists():
#         return {"error": "config not found"}
#     return load_yaml_config(path)





# class PreviewRequest(BaseModel):
#     tile_dir: str
#     bands: list[str] = ["B04", "B03", "B02"]
#     indices: list[str] = ["NDVI"]


# @app.post("/preview")
# def preview(req: PreviewRequest):
#     tile_dir = Path(req.tile_dir)
#     x, mapping = load_s2_stack(tile_dir, list(set(req.bands + ["B04", "B03", "B02"])) )
#     # RGB using B04,B03,B02
#     red = x[mapping["B04"]]
#     green = x[mapping["B03"]]
#     blue = x[mapping["B02"]]
#     def stretch(b):
#         p2, p98 = np.percentile(b, 2), np.percentile(b, 98)
#         return np.clip((b - p2) / (p98 - p2 + 1e-6), 0, 1)
#     rgb = np.stack([stretch(red), stretch(green), stretch(blue)], axis=-1)
#     rgb8 = (rgb * 255).astype(np.uint8)

#     # NDVI
#     if "NDVI" in req.indices:
#         x2, mapping2 = load_s2_stack(tile_dir, ["B08", "B04"])  # ensure mapping
#         nir = x2[mapping2["B08"]]
#         red = x2[mapping2["B04"]]
#         ndvi = (nir - red) / (nir + red + 1e-6)
#         ndvi_img = ((ndvi + 1) / 2 * 255).astype(np.uint8)
#         ndvi_b64 = np_to_png_b64(ndvi_img)
#     else:
#         ndvi_b64 = None

#     return {
#         "rgb_png": np_to_png_b64(rgb8),
#         "ndvi_png": ndvi_b64,
#         "shape": rgb8.shape,
#     }



# class PredictRequest(BaseModel):
#     tile_dir: str
#     checkpoint: Optional[str] = None


# @app.post("/predict")
# def predict(req: PredictRequest):
#     # Minimal: load checkpointed model if provided; otherwise error
#     ckpt = req.checkpoint
#     if ckpt is None or not Path(ckpt).exists():
#         return {"error": "checkpoint not found", "ok": False}

#     lit = LitClassifier.load_from_checkpoint(ckpt)
#     model = lit.model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)

#     # Load tile RGB bands default
#     x, mapping = load_s2_stack(Path(req.tile_dir), ["B02", "B03", "B04", "B08"])  # + NDVI
#     x = compute_indices(x, mapping, ["NDVI"])  # channels=5

#     patch_size = 256
#     stride = 256
#     H, W = x.shape[1], x.shape[2]
#     pred_map = np.full((H, W), 255, dtype=np.uint8)
#     ent_map = np.zeros((H, W), dtype=np.float32)
#     for r, c, p in extract_patches(x, patch_size, stride):
#         xt = torch.from_numpy(p).unsqueeze(0).to(device)
#         res = run_batch(model, xt)
#         cls = int(res.probs.argmax(axis=1)[0])
#         ent = float(res.entropy[0])
#         pred_map[r:r+patch_size, c:c+patch_size] = cls
#         ent_map[r:r+patch_size, c:c+patch_size] = ent

#     # Scale entropy to 0..255 for quick view
#     em = ent_map
#     em = (em - em.min()) / (em.max() - em.min() + 1e-6)
#     ent_png = np_to_png_b64((em * 255).astype(np.uint8))
#     pred_png = np_to_png_b64(pred_map.astype(np.uint8))
#     return {"ok": True, "pred_png": pred_png, "entropy_png": ent_png}


# class FinetuneRequest(BaseModel):
#     checkpoint: str
#     new_data_root: str = "data/processed/patches"
#     config_name: str = "option_c_supervised"


# def finetune_job(job_id: str, req: FinetuneRequest):
#     try:
#         cfg = load_yaml_config(Path("configs") / f"{req.config_name}.yaml")
#         res = finetune_from_checkpoint(req.checkpoint, cfg, req.new_data_root)
#         jobs[job_id].update({"status": "completed", **res})
#     except Exception as e:
#         jobs[job_id].update({"status": "failed", "error": str(e)})


# @app.post("/finetune")
# def start_finetune(req: FinetuneRequest):
#     job_id = f"job_{int(time.time())}"
#     jobs[job_id] = {"status": "running"}
#     t = threading.Thread(target=finetune_job, args=(job_id, req), daemon=True)
#     t.start()
#     return {"job_id": job_id}


# @app.get("/metrics")
# def get_metrics():
#     # Placeholder: return last job info
#     return jobs
