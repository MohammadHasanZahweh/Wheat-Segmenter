from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
from pathlib import Path
from shapely.geometry import Polygon
from rasterio.warp import transform_bounds


def tile_bounds_latlon(month_paths: Dict[int, str]) -> Tuple[float, float, float, float] | None:
    """Get bounds of a tile in EPSG:4326 (lon/lat)."""
    for _, p in month_paths.items():
        try:
            with rasterio.open(p) as ds:
                b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
            return b
        except Exception:
            continue
    return None


def bounds_to_polygon(b: Tuple[float, float, float, float]) -> Polygon:
    minx, miny, maxx, maxy = b
    return Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])


def load_tiles_index(ds) -> List[Dict[str, Any]]:
    """Create index of tiles with lat/lon bounds."""
    idx: List[Dict[str, Any]] = []
    for rec in ds.index:
        bounds_ll = tile_bounds_latlon(rec["month_paths"])
        if bounds_ll is None:
            continue
        idx.append(
            {
                "region": rec["region"],
                "tile_id": rec["tile_id"],
                "bounds": bounds_ll,
            }
        )
    return idx


def parse_regions(text: str) -> List[str] | None:
    tokens = [tok.strip() for tok in text.replace(",", " ").split()]
    regions = [tok for tok in tokens if tok]
    return regions or None


def cov_color(v: float) -> str:
    """Color for coverage: green (high) → red (low)."""
    v = max(0.0, min(1.0, float(v)))
    if v > 0.7:
        return "#00FF00"
    elif v > 0.5:
        return "#7FFF00"
    elif v > 0.3:
        return "#FFFF00"
    elif v > 0.1:
        return "#FFA500"
    else:
        return "#FF0000"
