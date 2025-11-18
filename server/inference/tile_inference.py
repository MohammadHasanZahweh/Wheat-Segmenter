import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Callable, List
import os
from server.config import DATA_PATH

months = [11,12,1,2,3,4,5,6,7,]
years = [2020,]
aois = [0,1,2,3,4]
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

# print(get_file_name(DATA_PATH,2020,11,0),get_file_name(DATA_PATH,2020,7,0))



def run_on_tile_one_year(
    year: int,
    aoi: int,
    process_fn: Callable[[np.ndarray], np.ndarray],
    out_path: str,
    patch_size: int = 256,
    stride: int = 256,
):
    """
    Read all images for one tile in one year, apply `process_fn`
    on patches, and save the output as a GeoTIFF.

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
    patch_size : int
        Spatial patch size (in pixels).
    stride : int
        Stride between patches (<= patch_size for overlap, == patch_size for no overlap).
    """

    # ------------------------------------------------------
    # 1) Get the file names for this tile/year
    # ------------------------------------------------------
    file_paths: List[str] = get_files_list(DATA_PATH, year, aoi)
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
    # 2) Prepare output raster (lazy init after first patch)
    # ------------------------------------------------------
    dst = None  # will be created after we see first out_patch

    # ------------------------------------------------------
    # 3) Iterate over patches
    # ------------------------------------------------------
    for row_start in range(0, height, stride):
        patch_h = min(patch_size, height - row_start)

        for col_start in range(0, width, stride):
            patch_w = min(patch_size, width - col_start)

            window = Window(col_start, row_start, patch_w, patch_h)

            # Read the same window from each time step, stack as (T, B, H, W)
            # Each src.read(window=window) -> (B, H, W)
            time_stack = np.stack(
                [src.read(window=window) for src in srcs], axis=0
            )  # shape: (T, B, H, W)

            # --------------------------------------------------
            # 4) Apply the user-provided function on this patch
            # --------------------------------------------------
            out_patch = process_fn(time_stack)  # np.ndarray

            # Normalize shape:
            #   - If 2D (H, W), add a band dimension => (1, H, W)
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

            # --------------------------------------------------
            # 5) Create output raster on first patch
            # --------------------------------------------------
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


# ----------------------------------------------------------
# Example usage (you will define your own process_fn)
# ----------------------------------------------------------
if __name__ == "__main__":
    def example_process_fn(patch_4d: np.ndarray) -> np.ndarray:
        """
        Example: simple temporal mean over the 9 dates,
        then sum over bands -> 1 output band.
        patch_4d: (T, B, H, W)
        """
        # Temporal mean: (T,B,H,W) -> (B,H,W)
        mean_time = patch_4d.mean(axis=0)
        # Sum over bands: (B,H,W) -> (H,W)
        summed = mean_time.sum(axis=0)
        return summed  # shape (H,W)

    run_on_tile_one_year(
        year=2020,
        aoi=0,                  # example tile id
        process_fn=example_process_fn, # replace with your model inference
        out_path="output_2020_T36SYJ.tif",
        patch_size=256,
        stride=256,
    )
