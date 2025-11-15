from __future__ import annotations
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


def rasterize_labels(
    label_shp: str,
    reference_raster: str,
    class_field: Optional[str] = None,
    background: int = 255,
) -> np.ndarray:
    """Rasterize polygons onto the reference raster grid.

    Args:
        label_shp: Path to shapefile (or GeoPackage/GeoJSON)
        reference_raster: Path to raster defining grid/transform/shape/CRS
        class_field: If None, burns value 1; else uses this column for class codes
        background: Nodata/ignore label
    Returns:
        2D array of integer labels aligned to reference raster
    """
    gdf = gpd.read_file(label_shp)
    with rasterio.open(reference_raster) as ref:
        transform = ref.transform
        out_shape = (ref.height, ref.width)
        gdf = gdf.to_crs(ref.crs)

    shapes = (
        (geom, int(row[class_field]) if class_field else 1)
        for _, row in gdf.iterrows()
        for geom in [row.geometry]
    )
    out = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        fill=background,
        transform=transform,
        dtype="uint8",
        all_touched=False,
    )
    return out


def patch_label_from_pixels(y_patch: np.ndarray, threshold: float = 0.6, ignore_value: int = 255) -> int | None:
    """Derive a patch-level label from per-pixel labels by majority threshold.

    Returns class id or None if ambiguous/insufficient coverage.
    """
    mask = y_patch != ignore_value
    if mask.sum() == 0:
        return None
    vals, counts = np.unique(y_patch[mask], return_counts=True)
    i = counts.argmax()
    maj_val, maj_ratio = int(vals[i]), counts[i] / mask.sum()
    if maj_ratio >= threshold:
        return maj_val
    return None

