from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict

from fastapi import FastAPI

import threading
import time
from uuid import uuid4

from .config import MODELS_PATH, TileDatasetConfig, TileTrainRequest, TrainingAlgorithm
from server.train import train_histgb, train_rf_baseline, train_svm_baseline, train_xgboost
from server.train.pixel_train import train_pixel_model

app = FastAPI(title="Wheat Mapping API")


def _dataset_kwargs(cfg: TileDatasetConfig) -> dict[str, Any]:
    return {
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


def _resolve_save_path(req: TileTrainRequest) -> str | None:
    if not req.save_model:
        return None
    if req.output_path:
        return str(Path(req.output_path))
    default_name = f"{req.job_name}_{req.algorithm.value}.joblib"
    return str((MODELS_PATH / default_name).resolve())


def _run_svm_job(req: TileTrainRequest) -> dict[str, Any]:
    params = {"svm_kernel": "rbf", "svm_C": 1.0, "svm_gamma": "scale"}
    params.update(req.model_params or {})
    gamma_raw = params.get("svm_gamma", "scale")
    try:
        gamma_val: str | float = float(gamma_raw)
    except (TypeError, ValueError):
        gamma_val = str(gamma_raw)
    cfg = train_svm_baseline.Config(
        **_dataset_kwargs(req.dataset),
        svm_kernel=str(params.get("svm_kernel", "rbf")),
        svm_C=float(params.get("svm_C", 1.0)),
        svm_gamma=gamma_val,
        save_model=_resolve_save_path(req),
    )
    return train_svm_baseline.train_and_eval(cfg)


def _run_rf_job(req: TileTrainRequest) -> dict[str, Any]:
    params = {"rf_estimators": 200, "rf_max_depth": None}
    params.update(req.model_params or {})
    depth = params.get("rf_max_depth", None)
    cfg = train_rf_baseline.Config(
        **_dataset_kwargs(req.dataset),
        rf_estimators=int(params.get("rf_estimators", 200)),
        rf_max_depth=None if depth in (None, 0, "0", "None") else int(depth),
        save_model=_resolve_save_path(req),
    )
    return train_rf_baseline.train_and_eval(cfg)


def _run_histgb_job(req: TileTrainRequest) -> dict[str, Any]:
    params = {
        "max_depth": 8,
        "max_iter": 400,
        "learning_rate": 0.05,
        "l2_regularization": 0.0,
    }
    params.update(req.model_params or {})
    depth = params.get("max_depth", 8)
    cfg = train_histgb.Config(
        **_dataset_kwargs(req.dataset),
        max_depth=None if depth in (None, 0, "0", "None") else int(depth),
        max_iter=int(params.get("max_iter", 400)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        l2_regularization=float(params.get("l2_regularization", 0.0)),
        save_model=_resolve_save_path(req),
    )
    return train_histgb.train_and_eval(cfg)


def _run_xgb_job(req: TileTrainRequest) -> dict[str, Any]:
    params = {
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    params.update(req.model_params or {})
    cfg = train_xgboost.Config(
        **_dataset_kwargs(req.dataset),
        n_estimators=int(params.get("n_estimators", 400)),
        max_depth=int(params.get("max_depth", 8)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        save_model=_resolve_save_path(req),
    )
    return train_xgboost.train_and_eval(cfg)


TRAINING_RUNNERS: Dict[TrainingAlgorithm, Callable[[TileTrainRequest], dict[str, Any]]] = {
    TrainingAlgorithm.SVM: _run_svm_job,
    TrainingAlgorithm.RANDOM_FOREST: _run_rf_job,
    TrainingAlgorithm.HISTOGRAM_GB: _run_histgb_job,
    TrainingAlgorithm.XGBOOST: _run_xgb_job,
}


def _run_tile_training(req: TileTrainRequest) -> dict[str, Any]:
    runner = TRAINING_RUNNERS.get(req.algorithm)
    if runner is None:
        raise ValueError(f"Unsupported algorithm {req.algorithm}")
    result = runner(req)
    result.setdefault("status", "completed")
    result["algorithm"] = req.algorithm.value
    result["job_name"] = req.job_name
    return result


jobs: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def train_job(job_id: str, payload: dict[str, Any]) -> None:
    try:
        req = TileTrainRequest(**payload)
        result = _run_tile_training(req)
        jobs[job_id].update(result)
        jobs[job_id]["status"] = "completed"
    except Exception as exc:  # pragma: no cover - surfaced via API
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
    payload = req.model_dump()
    thread = threading.Thread(target=train_job, args=(job_id, payload), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "running"}


@app.get("/train/status")
def train_status(id: str):
    return jobs.get(id, {"status": "unknown"})


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
