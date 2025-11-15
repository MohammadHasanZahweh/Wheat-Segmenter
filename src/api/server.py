from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi import Body
from pydantic import BaseModel
import threading
import time
import base64
import io
import numpy as np
from PIL import Image
from pathlib import Path

from src.data.sentinel2 import load_s2_stack, compute_indices
from src.data.tiling import extract_patches
from src.inference.service import run_batch
from src.train.train_supervised import train_from_config
from src.train.finetune import finetune_from_checkpoint
from src.data.dataset import PatchDataset
import torch

from src.utils.config import load_yaml_config


app = FastAPI(title="Wheat Mapping API")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/config/{name}")
def get_config(name: str) -> Dict:
    path = Path("configs") / f"{name}.yaml"
    if not path.exists():
        return {"error": "config not found"}
    return load_yaml_config(path)


def np_to_png_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class PreviewRequest(BaseModel):
    tile_dir: str
    bands: list[str] = ["B04", "B03", "B02"]
    indices: list[str] = ["NDVI"]


@app.post("/preview")
def preview(req: PreviewRequest):
    tile_dir = Path(req.tile_dir)
    x, mapping = load_s2_stack(tile_dir, list(set(req.bands + ["B04", "B03", "B02"])) )
    # RGB using B04,B03,B02
    red = x[mapping["B04"]]
    green = x[mapping["B03"]]
    blue = x[mapping["B02"]]
    def stretch(b):
        p2, p98 = np.percentile(b, 2), np.percentile(b, 98)
        return np.clip((b - p2) / (p98 - p2 + 1e-6), 0, 1)
    rgb = np.stack([stretch(red), stretch(green), stretch(blue)], axis=-1)
    rgb8 = (rgb * 255).astype(np.uint8)

    # NDVI
    if "NDVI" in req.indices:
        x2, mapping2 = load_s2_stack(tile_dir, ["B08", "B04"])  # ensure mapping
        nir = x2[mapping2["B08"]]
        red = x2[mapping2["B04"]]
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndvi_img = ((ndvi + 1) / 2 * 255).astype(np.uint8)
        ndvi_b64 = np_to_png_b64(ndvi_img)
    else:
        ndvi_b64 = None

    return {
        "rgb_png": np_to_png_b64(rgb8),
        "ndvi_png": ndvi_b64,
        "shape": rgb8.shape,
    }


class TrainRequest(BaseModel):
    config_name: str


jobs: dict[str, dict] = {}


def train_job(job_id: str, config_name: str):
    try:
        cfg = load_yaml_config(Path("configs") / f"{config_name}.yaml")
        res = train_from_config(cfg, data_root=cfg.get("data", {}).get("root", "data/processed/patches"))
        jobs[job_id].update({"status": "completed", **res})
    except Exception as e:
        jobs[job_id].update({"status": "failed", "error": str(e)})


@app.post("/train")
def start_train(req: TrainRequest):
    job_id = f"job_{int(time.time())}"
    jobs[job_id] = {"status": "running"}
    t = threading.Thread(target=train_job, args=(job_id, req.config_name), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/train/status")
def train_status(id: str):
    return jobs.get(id, {"status": "unknown"})


class PredictRequest(BaseModel):
    tile_dir: str
    checkpoint: Optional[str] = None


@app.post("/predict")
def predict(req: PredictRequest):
    # Minimal: load checkpointed model if provided; otherwise error
    ckpt = req.checkpoint
    if ckpt is None or not Path(ckpt).exists():
        return {"error": "checkpoint not found", "ok": False}

    from src.train.train_supervised import LitClassifier
    lit = LitClassifier.load_from_checkpoint(ckpt)
    model = lit.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load tile RGB bands default
    x, mapping = load_s2_stack(Path(req.tile_dir), ["B02", "B03", "B04", "B08"])  # + NDVI
    x = compute_indices(x, mapping, ["NDVI"])  # channels=5

    patch_size = 256
    stride = 256
    H, W = x.shape[1], x.shape[2]
    pred_map = np.full((H, W), 255, dtype=np.uint8)
    ent_map = np.zeros((H, W), dtype=np.float32)
    for r, c, p in extract_patches(x, patch_size, stride):
        xt = torch.from_numpy(p).unsqueeze(0).to(device)
        res = run_batch(model, xt)
        cls = int(res.probs.argmax(axis=1)[0])
        ent = float(res.entropy[0])
        pred_map[r:r+patch_size, c:c+patch_size] = cls
        ent_map[r:r+patch_size, c:c+patch_size] = ent

    # Scale entropy to 0..255 for quick view
    em = ent_map
    em = (em - em.min()) / (em.max() - em.min() + 1e-6)
    ent_png = np_to_png_b64((em * 255).astype(np.uint8))
    pred_png = np_to_png_b64(pred_map.astype(np.uint8))
    return {"ok": True, "pred_png": pred_png, "entropy_png": ent_png}


class FinetuneRequest(BaseModel):
    checkpoint: str
    new_data_root: str = "data/processed/patches"
    config_name: str = "option_c_supervised"


def finetune_job(job_id: str, req: FinetuneRequest):
    try:
        cfg = load_yaml_config(Path("configs") / f"{req.config_name}.yaml")
        res = finetune_from_checkpoint(req.checkpoint, cfg, req.new_data_root)
        jobs[job_id].update({"status": "completed", **res})
    except Exception as e:
        jobs[job_id].update({"status": "failed", "error": str(e)})


@app.post("/finetune")
def start_finetune(req: FinetuneRequest):
    job_id = f"job_{int(time.time())}"
    jobs[job_id] = {"status": "running"}
    t = threading.Thread(target=finetune_job, args=(job_id, req), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/metrics")
def get_metrics():
    # Placeholder: return last job info
    return jobs
