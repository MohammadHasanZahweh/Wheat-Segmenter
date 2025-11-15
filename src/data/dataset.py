from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import json
import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Simple dataset reading .npz patches with metadata JSON next to them.

    Expected layout:
      data/processed/patches/<split>/<tile>/<date>/<row>_<col>.npz
      and optional <row>_<col>.json for metadata.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
        label_key: Optional[str] = "label",
        stats_path: Optional[str | Path] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.paths = list(self.root.joinpath(split).rglob("*.npz"))
        self.label_key = label_key
        self._mean = None
        self._std = None
        if stats_path is not None and Path(stats_path).exists():
            import json
            stats = json.loads(Path(stats_path).read_text())
            self._mean = torch.tensor(stats["mean"], dtype=torch.float32)  # (C,)
            self._std = torch.tensor(stats["std"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        arr = np.load(path)["x"].astype(np.float32)
        meta_path = path.with_suffix(".json")
        y = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            y = meta.get(self.label_key, None)
        x = torch.from_numpy(arr)
        if self._mean is not None and self._std is not None:
            # per-channel normalize
            x = (x - self._mean[:, None, None]) / (self._std[:, None, None] + 1e-6)
        if self.transform:
            x = self.transform(x)
        if y is None:
            return x
        return x, torch.tensor(y, dtype=torch.long)
