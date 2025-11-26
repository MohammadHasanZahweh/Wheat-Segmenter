from pathlib import Path
from typing import Callable, List, Union, Mapping, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import windows
from rasterio.features import rasterize
import shapely
import os

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


def _normalize_polygons(polygons) -> List[PolygonLike]:
    """
    Accepts:
    - single shapely geometry or GeoJSON-like mapping
    - list/tuple of those
    - GeoPandas GeoDataFrame / GeoSeries (duck-typed via `.geometry`)
    Returns a list of geometries/mappings.
    """
    # GeoDataFrame / GeoSeries (duck-typing, no hard dependency on geopandas)
    if hasattr(polygons, "geometry"):
        geoms = list(polygons.geometry)
    # Already a list/sequence of polygons
    elif isinstance(polygons, (list, tuple)):
        geoms = list(polygons)
    # Single polygon-like object
    else:
        geoms = [polygons]

    # Filter out None
    geoms = [g for g in geoms if g is not None]

    if len(geoms) == 0:
        raise ValueError("No valid polygons provided.")

    return geoms


def _polygon_bounds(poly: PolygonLike):
    """
    Get bounds for a single polygon-like object.
    """
    try:
        return tuple(poly.bounds)  # shapely-like
    except AttributeError:
        # Assume GeoJSON-like mapping with "bbox" or compute from coordinates
        if isinstance(poly, Mapping) and "bbox" in poly:
            return tuple(poly["bbox"])
        elif isinstance(poly, Mapping) and "coordinates" in poly:
            coords = np.array(poly["coordinates"][0])
            minx, miny = coords.min(axis=0)
            maxx, maxy = coords.max(axis=0)
            return (minx, miny, maxx, maxy)
        else:
            raise ValueError("Could not get bounds from polygon-like object.")


def run_on_tile_one_year(
    base_path: Path,
    year: int,
    aoi: int,
    process_fn: Callable[[np.ndarray], np.ndarray],
    out_path: str,
    polygons,  # <- now supports single polygon, list of polygons, or GeoPandas GeoDataFrame/GeoSeries
    patch_size: int = 256,
    stride: int = 256,
):
    """
    Read all images for one tile in one year, apply `process_fn`
    on patches, and save the output as a *cropped* GeoTIFF.

    - Only run inference on the intersection between the raster
      and the given polygon(s).
    - Pixels outside the polygon(s) remain 0 in the output.
    - Output is cropped to the minimal bounding box of the
      polygon(s) ∩ tile.

    Parameters
    ----------
    year : int
        Year of interest.
    aoi : str
        AOI / tile identifier used by get_file_names.
    process_fn : callable
        Input:  patch_data -> np.ndarray of shape (T, B, H, W)
        Output: out_patch -> np.ndarray of shape:
                - (H, W)        or
                - (C_out, H, W)
    out_path : str
        Path to the output GeoTIFF.
    polygons :
        - single shapely geometry / GeoJSON-like mapping
        - list/sequence of those
        - GeoPandas GeoDataFrame / GeoSeries (same CRS as rasters)
    patch_size : int
        Spatial patch size (in pixels).
    stride : int
        Stride between patches (<= patch_size for overlap, == patch_size for no overlap).
    """

    # ------------------------------------------------------
    # 1) Normalize polygons & get the file names for this tile/year
    # ------------------------------------------------------
    geometries: List[PolygonLike] = _normalize_polygons(polygons)

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

    # ------------------------------------------------------
    # 2) Compute combined polygon bounds & crop window
    # ------------------------------------------------------
    raster_bounds = ref.bounds  # (left, bottom, right, top)

    # Combined bounds from all geometries
    all_bounds = np.array([_polygon_bounds(g) for g in geometries])
    minx = float(all_bounds[:, 0].min())
    miny = float(all_bounds[:, 1].min())
    maxx = float(all_bounds[:, 2].max())
    maxy = float(all_bounds[:, 3].max())
    poly_bounds_combined = (minx, miny, maxx, maxy)

    if not _bounds_intersect(raster_bounds, poly_bounds_combined):
        # No overlap at all
        for src in srcs:
            src.close()
        raise ValueError("Polygon(s) do not intersect with raster.")

    # Clamp polygon bounds to raster bounds to get the intersection bbox in world coords
    r_left, r_bottom, r_right, r_top = raster_bounds
    inter_left = max(minx, r_left)
    inter_bottom = max(miny, r_bottom)
    inter_right = min(maxx, r_right)
    inter_top = min(maxy, r_top)

    if inter_left >= inter_right or inter_bottom >= inter_top:
        for src in srcs:
            src.close()
        raise ValueError("Polygon(s) intersection with raster is empty.")

    # Convert intersection bounds to pixel indices
    # Top-left pixel: (inter_left, inter_top)
    # Bottom-right pixel: (inter_right, inter_bottom)
    row_min, col_min = ref.index(inter_left, inter_top)
    row_max, col_max = ref.index(inter_right, inter_bottom)

    row_start = max(0, min(row_min, row_max))
    row_stop = min(height, max(row_min, row_max) + 1)
    col_start = max(0, min(col_min, col_max))
    col_stop = min(width, max(col_min, col_max) + 1)

    if row_start >= row_stop or col_start >= col_stop:
        for src in srcs:
            src.close()
        raise ValueError("Computed crop window is empty.")

    crop_window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
    crop_height = int(crop_window.height)
    crop_width = int(crop_window.width)
    crop_transform = windows.transform(crop_window, ref.transform)

    # ------------------------------------------------------
    # 3) Prepare output raster (lazy init after first patch)
    # ------------------------------------------------------
    dst = None  # will be created after we see first out_patch

    # ------------------------------------------------------
    # 4) Iterate over patches, but only within crop_window & where polygons intersect
    # ------------------------------------------------------
    for row0 in range(row_start, row_stop, stride):
        patch_h = min(patch_size, row_stop - row0)

        for col0 in range(col_start, col_stop, stride):
            patch_w = min(patch_size, col_stop - col0)

            # Window in source (full raster) coordinates
            src_window = Window(col0, row0, patch_w, patch_h)

            # Window bounds in raster CRS
            win_bounds = windows.bounds(src_window, ref.transform)  # (left, bottom, right, top)

            # Quick reject via bbox intersection (combined polygon bounds)
            if not _bounds_intersect(win_bounds, poly_bounds_combined):
                continue

            # More precise check: rasterize polygons into this window
            win_transform = windows.transform(src_window, ref.transform)

            shapes = []
            for g in geometries:
                # rasterio.features.rasterize accepts either shapely or GeoJSON-like
                shapes.append((g, 1))

            mask = rasterize(
                shapes,
                out_shape=(patch_h, patch_w),
                transform=win_transform,
                fill=0,
                dtype="uint8",
                all_touched=False,
            )

            # If polygons do not cover any pixel in this window, skip
            if mask.max() == 0:
                continue

            # --------------------------------------------------
            # 5) Read data and apply user-provided function
            # --------------------------------------------------
            time_stack = np.stack(
                [src.read(window=src_window) for src in srcs], axis=0
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

            # Apply polygon mask: keep predictions only inside polygons
            mask_bool = mask.astype(bool)
            if not mask_bool.all():
                out_patch[:, ~mask_bool] = 0

            out_patch = out_patch.astype(np.uint8)

            # --------------------------------------------------
            # 6) Create output raster on first patch (cropped)
            # --------------------------------------------------
            if dst is None:
                out_meta = ref.meta.copy()
                out_meta.update(
                    {
                        "count": out_c,
                        "height": crop_height,
                        "width": crop_width,
                        "dtype": out_patch.dtype,
                        "transform": crop_transform,
                    }
                )
                print(out_meta)
                dst = rasterio.open(out_path, "w", **out_meta)

            # --------------------------------------------------
            # 7) Write patch to correct window in *cropped* output
            # --------------------------------------------------
            # Destination window is offset relative to crop_window
            dst_row0 = row0 - row_start
            dst_col0 = col0 - col_start
            dst_window = Window(dst_col0, dst_row0, patch_w, patch_h)

            dst.write(out_patch, window=dst_window)

    # ------------------------------------------------------
    # 8) Clean up
    # ------------------------------------------------------
    if dst is not None:
        dst.close()
        print(f"Saved output to: {out_path}")
    else:
        print("No patches intersected the provided polygon(s); no output written.")

    for src in srcs:
        src.close()
