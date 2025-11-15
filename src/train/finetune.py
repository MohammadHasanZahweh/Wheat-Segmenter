from __future__ import annotations
from typing import Dict

import torch
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import mlflow

from .train_supervised import LitClassifier


def finetune_from_checkpoint(checkpoint: str, cfg: Dict, new_data_root: str) -> Dict[str, str]:
    """Fine-tune an existing classifier on new data.

    Loads checkpoint, reduces LR, and trains for a few epochs.
    """
    lit = LitClassifier.load_from_checkpoint(checkpoint)
    # Reduce LR for finetune if provided
    if hasattr(lit, "lr"):
        lit.lr = cfg.get("train", {}).get("lr", 5e-4)

    # TODO: build DataLoader from new_data_root (similar to train_from_config)
    # from src.data.dataset import PatchDataset
    # stats_path = cfg.get("data", {}).get("stats_path", None)
    # train_ds = PatchDataset(root=new_data_root, split=cfg.get("data", {}).get("train_split", "train"), stats_path=stats_path)
    # val_ds = PatchDataset(root=new_data_root, split=cfg.get("data", {}).get("val_split", "val"), stats_path=stats_path)
    # train_loader = DataLoader(train_ds, batch_size=cfg.get("train", {}).get("batch_size", 32), shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=cfg.get("train", {}).get("batch_size", 32), shuffle=False)

    ckpt = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
    es = EarlyStopping(monitor="val_acc", mode="max", patience=cfg.get("train", {}).get("early_stopping_patience", 3))

    exp_name = cfg.get("name", "finetune") + "_finetune"
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "file:./runs/mlruns")
    mlf_logger = MLFlowLogger(experiment_name=exp_name, tracking_uri=tracking_uri)
    mlf_logger.log_hyperparams({"checkpoint": checkpoint, **cfg})

    trainer = Trainer(max_epochs=cfg.get("train", {}).get("epochs", 5), accelerator="auto", callbacks=[ckpt, es], logger=mlf_logger)
    # trainer.fit(lit, train_loader, val_loader)

    best_path = ckpt.best_model_path or checkpoint
    try:
        if best_path:
            mlf_logger.experiment.log_artifact(mlf_logger.run_id, best_path)
    except Exception:
        pass
    return {"checkpoint": best_path, "mlflow_run_id": mlf_logger.run_id, "mlflow_experiment_id": mlf_logger.experiment_id}
