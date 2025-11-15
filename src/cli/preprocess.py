from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.data.sentinel2 import load_s2_stack, compute_indices, mask_clouds
from src.data.tiling import extract_patches
from src.data.labels import rasterize_labels, patch_label_from_pixels


def percentile_stretch(arr: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    lo = np.percentile(arr, pmin)
    hi = np.percentile(arr, pmax)
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


def find_tile_dirs(s2_root: Path) -> List[Path]:
    # Expect structure: data/raw/sentinel2/<tile>/<date>/B02.tif
    paths = []
    for p in s2_root.rglob("B02.tif"):
        paths.append(p.parent)
    return sorted(paths)


def simple_spatial_split(paths: List[Path], ratios=(0.7, 0.15, 0.15)) -> Dict[Path, str]:
    # Deterministic split by hashing tile path string
    assign = {}
    for d in paths:
        h = abs(hash(str(d))) % 1000 / 1000.0
        if h < ratios[0]:
            assign[d] = "train"
        elif h < ratios[0] + ratios[1]:
            assign[d] = "val"
        else:
            assign[d] = "test"
    return assign


def accumulate_stats(stats: Dict[str, np.ndarray], x: np.ndarray, valid_mask: Optional[np.ndarray] = None):
    # x: (C,H,W); valid_mask: (H,W) boolean
    C = x.shape[0]
    if valid_mask is None:
        m = np.ones(x.shape[1:], dtype=bool)
    else:
        m = valid_mask
    count = m.sum()
    if count == 0:
        return
    xm = x[:, m]
    stats["sum"] += xm.sum(axis=1)
    stats["sumsq"] += (xm ** 2).sum(axis=1)
    stats["count"] += count


def run(
    s2_root: Path,
    labels_path: Path,
    output_root: Path,
    class_field: Optional[str],
    patch_size: int,
    stride: int,
    bands: List[str],
    indices: List[str],
    label_threshold: float,
):
    output_root.mkdir(parents=True, exist_ok=True)
    paths = find_tile_dirs(s2_root)
    assign = simple_spatial_split(paths)

    # Stats containers for train split
    stats = {"sum": None, "sumsq": None, "count": 0}

    for tile_dir in tqdm(paths, desc="Tiles"):
        try:
            x, mapping = load_s2_stack(tile_dir, bands)
            if indices:
                x = compute_indices(x, mapping, indices)
            scl = tile_dir / "SCL.tif"
            valid = mask_clouds(x, scl)
            # zero-out invalid pixels (keep mask for stats)
            x[:, ~valid] = 0
        except Exception as e:
            print(f"Failed to load {tile_dir}: {e}")
            continue

        # Prepare label raster
        ref_raster = tile_dir / "B02.tif"
        y = rasterize_labels(str(labels_path), str(ref_raster), class_field=class_field, background=255)

        split = assign[tile_dir]
        out_dir = output_root / split / tile_dir.relative_to(s2_root)
        out_dir.mkdir(parents=True, exist_ok=True)

        # init stats shape after channels known
        if stats["sum"] is None:
            C = x.shape[0]
            stats["sum"] = np.zeros(C, dtype=np.float64)
            stats["sumsq"] = np.zeros(C, dtype=np.float64)

        for r, c, patch in extract_patches(x, patch_size=patch_size, stride=stride):
            y_patch = y[r : r + patch_size, c : c + patch_size]
            if y_patch.shape[0] != patch_size or y_patch.shape[1] != patch_size:
                continue
            plabel = patch_label_from_pixels(y_patch, threshold=label_threshold, ignore_value=255)
            if plabel is None:
                continue
            # Save NPZ and metadata JSON
            name = f"{r}_{c}"
            np.savez_compressed(out_dir / f"{name}.npz", x=patch.astype(np.float32))
            meta = {
                "tile_dir": str(tile_dir),
                "row": r,
                "col": c,
                "label": int(plabel),
                "bands": bands,
                "indices": indices,
            }
            (out_dir / f"{name}.json").write_text(json.dumps(meta))

            if split == "train":
                valid_patch = (y_patch != 255)
                accumulate_stats(stats, patch, valid_patch)

    # finalize stats
    if stats["count"] > 0:
        mean = (stats["sum"] / stats["count"]).tolist()
        var = (stats["sumsq"] / stats["count"]) - np.array(mean) ** 2
        std = np.sqrt(np.maximum(var, 1e-12)).tolist()
        out_stats = {"mean": mean, "std": std}
        (output_root.parent / "stats.json").write_text(json.dumps(out_stats, indent=2))


def main():
    p = argparse.ArgumentParser(description="Preprocess Sentinel-2 and labels into patch dataset")
    p.add_argument("--s2_root", type=Path, default=Path("data/raw/sentinel2"))
    p.add_argument("--labels", type=Path, required=True, help="Path to shapefile/GeoJSON with labels")
    p.add_argument("--class_field", type=str, default=None, help="Column with class codes (None=binary=1)")
    p.add_argument("--out_root", type=Path, default=Path("data/processed/patches"))
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--bands", type=str, nargs="+", default=["B02", "B03", "B04", "B08"]) 
    p.add_argument("--indices", type=str, nargs="*", default=["NDVI"])
    p.add_argument("--label_threshold", type=float, default=0.6)
    args = p.parse_args()

    run(
        s2_root=args.s2_root,
        labels_path=args.labels,
        output_root=args.out_root,
        class_field=args.class_field,
        patch_size=args.patch_size,
        stride=args.stride,
        bands=args.bands,
        indices=args.indices,
        label_threshold=args.label_threshold,
    )


if __name__ == "__main__":
    main()

