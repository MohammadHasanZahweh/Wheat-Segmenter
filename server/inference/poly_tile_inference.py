from pathlib import Path
from typing import Callable, List, Union, Mapping

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import windows
from rasterio.features import rasterize
import shapely
import os

# You can use shapely.geometry.Polygon, but we only assume it has a .bounds and
# geo-interface (which shapely provides), or is a GeoJSON-like mapping.
PolygonLike = Union[Mapping, "shapely.geometry.base.BaseGeometry"]

def get_file_name(base_path, year, month, aoi):
    """
    Returns a list of file paths (9 GeoTIFFs with 13 bands each)
    for a single tile / AOI and a given year.
    """
    if month > 10:
        path = f"year_{year-1}/aoi_0_{aoi}/month_{month:02}"
    else:
        path = f"year_{year}/aoi_0_{aoi}/month_{month:02}"
    
    path = os.path.join(base_path, path)
    path = os.path.join(path, os.listdir(path)[0], "response.tiff")
    return path

def get_files_list(base_path, year, aoi, months=[11,12,1,2,3,4,5,6,7]):
    return [get_file_name(base_path, year, month, aoi) for month in months]

def _bounds_intersect(b1, b2) -> bool:
    """
    Quick bbox intersection check.
    b = (left, bottom, right, top)
    """
    left1, bottom1, right1, top1 = b1
    left2, bottom2, right2, top2 = b2
    return not (right1 <= left2 or right2 <= left1 or top1 <= bottom2 or top2 <= bottom1)


def run_on_tile_one_year(
    base_path: Path,
    year: int,
    aoi: int,
    process_fn: Callable[[np.ndarray], np.ndarray],
    out_path: str,
    polygon: PolygonLike,
    patch_size: int = 256,
    stride: int = 256,
):
    """
    Read all images for one tile in one year, apply `process_fn`
    on patches, and save the output as a GeoTIFF.

    In addition, only run inference on the intersection between the
    raster and a given polygon. Pixels outside the polygon remain 0
    in the output.

    Parameters
    ----------
    year : int
        Year of interest.
    aoi : str
        AOI / tile identifier used by get_file_names.
    process_fn : callable
        Function provided by the user.
        Input:  patch_data -> np.ndarray of shape (T, B, H, W)
                where T = number of time steps (e.g. 9), B = bands (13)
        Output: out_patch -> np.ndarray of shape:
                - (H, W)        or
                - (C_out, H, W)  (C_out = number of output channels)
    out_path : str
        Path to the output GeoTIFF.
    polygon : PolygonLike
        Polygon geometry in the *same CRS* as the rasters.
        Can be a shapely geometry or a GeoJSON-like mapping.
    patch_size : int
        Spatial patch size (in pixels).
    stride : int
        Stride between patches (<= patch_size for overlap, == patch_size for no overlap).
    """

    # ------------------------------------------------------
    # 1) Get the file names for this tile/year
    # ------------------------------------------------------
    file_paths: List[str] = get_files_list(base_path, year, aoi)
    if len(file_paths) == 0:
        raise ValueError(f"No files found for year={year}, aoi={aoi}")

    # Open all images (9 time steps, each with 13 bands)
    srcs = [rasterio.open(fp) for fp in file_paths]

    # Check basic consistency (CRS, transform, size, band count)
    ref = srcs[0]
    height, width = ref.height, ref.width
    num_bands = ref.count
    num_times = len(srcs)

    for src in srcs[1:]:
        if (src.height != height) or (src.width != width):
            raise ValueError("All images must have the same spatial size.")
        if src.count != num_bands:
            raise ValueError("All images must have the same number of bands.")
        if src.crs != ref.crs or src.transform != ref.transform:
            raise ValueError("All images must have the same CRS and transform.")

    # Quick overall bbox intersection check between polygon and raster
    raster_bounds = ref.bounds  # (left, bottom, right, top)
    try:
        poly_bounds = polygon.bounds  # shapely-like
    except AttributeError:
        # Assume GeoJSON-like mapping with "bbox" or compute from coordinates
        if "bbox" in polygon:
            poly_bounds = tuple(polygon["bbox"])
        else:
            # Very basic bbox from coordinates (GeoJSON, assuming Polygon)
            coords = np.array(polygon["coordinates"][0])
            minx, miny = coords.min(axis=0)
            maxx, maxy = coords.max(axis=0)
            poly_bounds = (minx, miny, maxx, maxy)

    if not _bounds_intersect(raster_bounds, poly_bounds):
        # No overlap at all
        for src in srcs:
            src.close()
        raise ValueError("Polygon does not intersect with raster.")

    # ------------------------------------------------------
    # 2) Prepare output raster (lazy init after first patch)
    # ------------------------------------------------------
    dst = None  # will be created after we see first out_patch

    # ------------------------------------------------------
    # 3) Iterate over patches, but only process where polygon intersects
    # ------------------------------------------------------
    for row_start in range(0, height, stride):
        patch_h = min(patch_size, height - row_start)

        for col_start in range(0, width, stride):
            patch_w = min(patch_size, width - col_start)

            window = Window(col_start, row_start, patch_w, patch_h)

            # Window bounds in raster CRS
            win_bounds = windows.bounds(window, ref.transform)  # (left, bottom, right, top)

            # Quick reject via bbox intersection
            if not _bounds_intersect(win_bounds, poly_bounds):
                continue

            # More precise check: rasterize polygon into this window to see if any pixels are inside
            win_transform = windows.transform(window, ref.transform)

            mask = rasterize(
                [(polygon, 1)],
                out_shape=(patch_h, patch_w),
                transform=win_transform,
                fill=0,
                dtype="uint8",
                all_touched=False,
            )

            # If polygon does not cover any pixel in this window, skip
            if mask.max() == 0:
                continue

            # --------------------------------------------------
            # 4) Read data and apply user-provided function
            # --------------------------------------------------
            time_stack = np.stack(
                [src.read(window=window) for src in srcs], axis=0
            )  # shape: (T, B, H, W)

            out_patch = process_fn(time_stack)  # np.ndarray

            # Normalize shape:
            if out_patch.ndim == 2:
                out_patch = out_patch[np.newaxis, ...]  # (1, H, W)
            elif out_patch.ndim != 3:
                raise ValueError(
                    f"process_fn must return (H,W) or (C,H,W). Got shape {out_patch.shape}"
                )

            out_c, out_h, out_w = out_patch.shape
            if (out_h != patch_h) or (out_w != patch_w):
                raise ValueError(
                    f"process_fn output spatial size ({out_h},{out_w}) "
                    f"does not match patch size ({patch_h},{patch_w})."
                )

            # Apply polygon mask: keep predictions only inside polygon
            # mask: (H, W) with values 0/1
            mask_bool = mask.astype(bool)
            if not mask_bool.all():
                # For every output channel, zero out pixels outside the polygon
                out_patch[:, ~mask_bool] = 0

            # --------------------------------------------------
            # 5) Create output raster on first patch
            # --------------------------------------------------
            out_patch = out_patch.astype(np.uint8)
            if dst is None:
                out_meta = ref.meta.copy()
                out_meta.update(
                    {
                        "count": out_c,
                        "height": height,
                        "width": width,
                        "dtype": out_patch.dtype,
                    }
                )
                print(out_meta)
                dst = rasterio.open(out_path, "w", **out_meta)

            # --------------------------------------------------
            # 6) Write patch to correct window in output
            # --------------------------------------------------
            dst.write(out_patch, window=window)

    # ------------------------------------------------------
    # 7) Clean up
    # ------------------------------------------------------
    if dst is not None:
        dst.close()
    for src in srcs:
        src.close()
    print(f"Saved output to: {out_path}")
