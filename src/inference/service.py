from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from src.models.classifier import PatchClassifier


@dataclass
class InferenceResult:
    logits: np.ndarray  # (N, C)
    probs: np.ndarray   # (N, C)
    entropy: np.ndarray # (N,)


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    return ent


def run_batch(model: PatchClassifier, x: torch.Tensor) -> InferenceResult:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        ent = softmax_entropy(logits)
    return InferenceResult(
        logits=logits.cpu().numpy(),
        probs=probs.cpu().numpy(),
        entropy=ent.cpu().numpy(),
    )

