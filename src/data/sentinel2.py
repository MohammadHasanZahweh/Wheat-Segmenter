from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


S2_10M = ["B02", "B03", "B04", "B08"]
S2_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]


def _open_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1)


def resample_to(reference: Path, source: Path) -> np.ndarray:
    with rasterio.open(reference) as ref:
        dst_shape = (ref.height, ref.width)
        dst_transform = ref.transform
        dst = np.empty(dst_shape, dtype=np.float32)
        with rasterio.open(source) as src:
            reproject(
                source=src.read(1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=ref.crs,
                resampling=Resampling.bilinear,
            )
    return dst


def load_s2_stack(tile_dir: str | Path, bands: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load and stack Sentinel-2 bands; resample 20m to 10m if needed.

    Assumes files named like B02.tif, B03.tif, ... in tile_dir.

    Returns:
        array (C,H,W) and band index mapping
    """
    tile_dir = Path(tile_dir)
    ref_path = tile_dir / "B02.tif"
    with rasterio.open(ref_path) as ref:
        H, W = ref.height, ref.width
    stack = []
    mapping: Dict[str, int] = {}
    for i, b in enumerate(bands):
        path = tile_dir / f"{b}.tif"
        if b in S2_10M:
            arr = _open_band(path)
        else:
            arr = resample_to(ref_path, path)
        stack.append(arr.astype(np.float32))
        mapping[b] = i
    x = np.stack(stack, axis=0)
    return x, mapping


def compute_indices(x: np.ndarray, mapping: Dict[str, int], indices: List[str]) -> np.ndarray:
    """Compute spectral indices and stack as extra channels.

    Supports: NDVI only for now.
    """
    extras = []
    for name in indices:
        if name.upper() == "NDVI":
            nir = x[mapping["B08"]]
            red = x[mapping["B04"]]
            ndvi = (nir - red) / (nir + red + 1e-6)
            extras.append(ndvi.astype(np.float32))
        else:
            raise NotImplementedError(f"Index {name} not implemented")
    if not extras:
        return x
    return np.concatenate([x, np.stack(extras, axis=0)], axis=0)


def mask_clouds(x: np.ndarray, scl_path: str | Path) -> np.ndarray:
    """Return a boolean mask of valid (non-cloud) pixels using Sentinel-2 SCL.

    Masks clouds, cloud shadows, cirrus, and saturated/defective pixels.
    """
    cloud_classes = {3, 8, 9, 10, 11}
    with rasterio.open(scl_path) as ds:
        scl = ds.read(1)
    valid = ~np.isin(scl, list(cloud_classes))
    # Broadcast to all channels: (C,H,W) & (H,W)
    return valid
