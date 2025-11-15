from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import mlflow

from src.data.dataset import PatchDataset
from src.models.encoders import SimpleMSResNetEncoder
from src.models.heads import MLPHead
from src.models.classifier import PatchClassifier


class LitClassifier(LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def build_supervised_model(in_channels: int, num_classes: int) -> nn.Module:
    encoder = SimpleMSResNetEncoder(in_channels=in_channels, out_dim=256)
    head = MLPHead(in_dim=256, hidden_dim=256, out_dim=num_classes)
    return PatchClassifier(encoder, head)


def train_from_config(cfg: Dict, data_root: str = "data/processed/patches") -> Dict[str, str]:
    """Train a supervised classifier using config values.

    Returns:
        Path to best checkpoint.
    """
    in_channels = cfg.get("model", {}).get("in_channels", 5)
    num_classes = cfg.get("num_classes", 2)
    epochs = cfg.get("train", {}).get("epochs", 10)
    batch_size = cfg.get("train", {}).get("batch_size", 32)
    lr = cfg.get("train", {}).get("lr", 1e-3)
    weight_decay = cfg.get("train", {}).get("weight_decay", 1e-4)
    num_workers = cfg.get("train", {}).get("num_workers", 4)
    patience = cfg.get("train", {}).get("early_stopping_patience", 5)

    model = build_supervised_model(in_channels, num_classes)
    lit = LitClassifier(model, lr=lr, weight_decay=weight_decay)

    stats_path = cfg.get("data", {}).get("stats_path", None)
    train_ds = PatchDataset(root=data_root, split=cfg.get("data", {}).get("train_split", "train"), stats_path=stats_path)
    val_ds = PatchDataset(root=data_root, split=cfg.get("data", {}).get("val_split", "val"), stats_path=stats_path)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    ckpt = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
    es = EarlyStopping(monitor="val_acc", mode="max", patience=patience)

    # MLflow logger
    exp_name = cfg.get("name", "option_c_supervised")
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "file:./runs/mlruns")
    mlf_logger = MLFlowLogger(experiment_name=exp_name, tracking_uri=tracking_uri)
    # Log hyperparameters
    mlf_logger.log_hyperparams(cfg)

    trainer = Trainer(max_epochs=epochs, callbacks=[ckpt, es], accelerator="auto", logger=mlf_logger)
    trainer.fit(lit, train_loader, val_loader)

    best_path = ckpt.best_model_path or ""
    # Log best checkpoint as artifact if available
    try:
        if best_path:
            mlf_logger.experiment.log_artifact(mlf_logger.run_id, best_path)
    except Exception:
        pass

    return {"checkpoint": best_path, "mlflow_run_id": mlf_logger.run_id, "mlflow_experiment_id": mlf_logger.experiment_id}
