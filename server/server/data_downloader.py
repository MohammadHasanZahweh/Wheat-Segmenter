import geopandas as gpd
import requests
import calendar
from sentinelhub import (
    CRS,
    BBox,
    Geometry,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
    BBoxSplitter
)

from sentinelhub import SHConfig
import matplotlib.pyplot as plt

from shapely.geometry import Polygon,MultiPolygon

import numpy as np
import os
import rasterio as rio
from tqdm import tqdm
from math import ceil


from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import sentinelhub
print(sentinelhub.__version__)

# Your client credentials
# Mohado5@gmail.com
client_id = 'sh-2296027c-f67a-422c-9277-0a2486f1e173'
client_secret = 'PzmZHPoKqoIq2hdjQ50GsITtCDdMHL80'

# Create a session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                          client_secret=client_secret)

# All requests using this session will have an access token automatically added
#resp = oauth.get("...")
#print(resp.content)

# #Configure
config = SHConfig()

config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_client_id = client_id
config.sh_client_secret = client_secret
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.save("cdse")
 
# !/home/jamada/jupyterlab/crop/bin/sentinelhub.config --profile cdse --show


evalscript_all_bands = """
function setup() {
  return {
    input: [{
      bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
      units: "DN"
    }],
    output: {
      id: "default",
      bands: 13,
      sampleType: SampleType.UINT16
    }
  }
}

function evaluatePixel(sample) {
    return [ sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B10, sample.B11, sample.B12]
}
"""

def split_aoi(geom_poly,crs,resolution):
    MAX_SIZE = 2500
    geom = Geometry(geometry=geom_poly,crs=crs)
    bbox = geom.bbox
    size = bbox_to_dimensions(bbox, resolution=resolution)

    if max(size) <= 2500:
        geoms = [geom]
        bboxes = [bbox]
        sizes = [size]
    
    else:
        w,h = size

        n = ceil(w / MAX_SIZE) 
        m = ceil(h / MAX_SIZE)
        grid = (n,m)

        bbox_splitter = BBoxSplitter(
            shape_list=[geom_poly],
            crs = crs,
            split_shape=grid,
            reduce_bbox_sizes=True
            
        )
        
        bboxes, info_list = bbox_splitter._make_split()
        bbox_polys = [Polygon(x.get_polygon()) for x in bboxes]
        geoms = [Geometry(geometry=bbox.intersection(geom_poly),crs=crs) for bbox in bbox_polys]
        sizes = [bbox_to_dimensions(bbox, resolution=resolution) for bbox in bboxes]
        #geoms = [bbox.intersection(geom_poly) for bbox in bbox_polys]
        #gdf = gpd.GeoDataFrame({'geometry':geoms},crs=str(crs))
        #gdf.plot()
        

    return geoms,bboxes,sizes

# years =[2019,2020]#[2015,2016,2017,2018,2019]#[2020,2021,2022]#[2022]#[2021] #[2016,2017,2018,2019,2020],
# months = ["11","12"]#['01','02','03','04','05','06','07']#'06','07','11','12']
def download_files(aois_gdf, years,  out_dir = "data/myregion/download"):
    geom = aois_gdf['geometry'][0]
    bbox_splitter = BBoxSplitter(
        shape_list=[geom],
        crs = aois_gdf.crs,
        split_shape=5,
        reduce_bbox_sizes=True
        )

    bbox_list, info_list = bbox_splitter._make_split()

    bboxes = [Polygon(x.get_polygon()) for x in bbox_list]

    bbox_aois_gdf = gpd.GeoDataFrame({'geometry':bboxes},crs=aois_gdf.crs)
    bbox_aois_gdf.plot()

    starting_from_aoi=0
    crs = CRS.WGS84 #aois_gdf.crs
    resolution = 10
    for i,row in aois_gdf.iterrows():

        geom_poly = row['geometry']

        geoms,bboxes,sizes = split_aoi(geom_poly,crs,resolution)

        #print(geoms,bboxes,sizes)

        for j in range(starting_from_aoi,len(geoms)):

            #if j < 3:
            #    continue
            # if first:
            #     first=False
            #     continue

            aoi_id = f'aoi_{i}_{j}'

            bbox = bboxes[j]
            geom = geoms[j]
            size = sizes[j]

            for index,year in enumerate(years):
                if index==0:
                    months=["11","12"]
                elif index==len(years)-1:
                    months=['01','02','03','04','05','06','07']
                else:
                    months = ['01','02','03','04','05','06','07','11','12']

                month_loader = tqdm(months)

                for month in month_loader: 

                    _,lastday = calendar.monthrange(year, int(month))

                    #print(f'{year}-{month}-{lastday}')
                    out_dir_now = f'{out_dir}/year_{year}/{aoi_id}/month_{month}'
                    month_loader.set_description(f'Requesting Tiff Images for {aoi_id} - Year{year} - Month{month} ...')
                    os.makedirs(out_dir_now,exist_ok=True)
                    request_all_bands = SentinelHubRequest(
                        evalscript=evalscript_all_bands,
                        input_data=[
                            SentinelHubRequest.input_data(
                                data_collection=DataCollection.SENTINEL2_L1C.define_from("cdse_sentinel2_l1c", service_url=config.sh_base_url ),
                                time_interval=(f'{year}-{month}-1', f'{year}-{month}-{lastday}'),
                                mosaicking_order=MosaickingOrder.LEAST_CC,
                            )
                        ],
                        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                        #bbox=bbox,
                        geometry=geom,
                        size=size,
                        config=config,
                        data_folder = out_dir_now

                    )

                    all_bands_imgs = request_all_bands.get_data(save_data=True)
