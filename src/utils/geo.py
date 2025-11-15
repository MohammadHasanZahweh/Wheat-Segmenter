from __future__ import annotations
from typing import Tuple
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS


def read_meta(path: str) -> Tuple[CRS, Affine, Tuple[int, int]]:
    """Read CRS, transform and shape from a raster.

    Args:
        path: Path to raster.
    Returns:
        (crs, transform, (height, width))
    """
    with rasterio.open(path) as ds:
        return ds.crs, ds.transform, (ds.height, ds.width)

