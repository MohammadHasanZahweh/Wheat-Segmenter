import rasterio
from rasterio.mask import mask
import geopandas as gpd
from server.server.config import DATA_PATH
import numpy as np


def crop_bands_432(year, month, polygon_path, output_path):
    """
    Crop a 13-band GeoTIFF using a polygon and save only bands [4,3,2].

    :param image_path: path to input 13-band TIFF
    :param polygon_path: path to shapefile/geojson containing polygon
    :param output_path: path to save resulting TIFF
    """

    # ----------------------------
    # 1) Read polygon geometry
    # ----------------------------
    gdf = gpd.read_file(polygon_path)
    geoms = gdf.geometry.values  # list of shapely polygons

    
    # ----------------------------
    # 2) Open image
    # ----------------------------
    image_path = DATA_PATH / "Lebanon/merge_data"/f"year_{year}_month_{month}.tiff"
    with rasterio.open(image_path) as src:
        
        # Bands to keep (Sentinel-2: 4=Red, 3=Green, 2=Blue)
        selected_bands = [4, 3, 2]

        # ----------------------------
        # 3) Mask the image with polygon
        # ----------------------------
        out_image, out_transform = mask(
            src,
            geoms,
            crop=True,
            all_touched=True,
            indexes=selected_bands
        )
        print(out_image.max(), out_image.min(), out_image.mean())
        out_image = (out_image/20).clip(max=255).astype(np.uint8)

        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width":  out_image.shape[2],
            "transform": out_transform,
            "count": len(selected_bands),
            "dtype":out_image.dtype,
        })

        # ----------------------------
        # 4) Save result
        # ----------------------------
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"[✓] Saved cropped RGB bands to: {output_path}")
