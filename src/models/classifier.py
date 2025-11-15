from __future__ import annotations
import torch
import torch.nn as nn


class PatchClassifier(nn.Module):
    """Encoder + classification head wrapper."""

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        logits = self.head(feats)
        return logits

