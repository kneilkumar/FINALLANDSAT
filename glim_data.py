import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from bmi_topography import Topography

station_ids = pd.read_csv("submission_template.csv")
station_ids = station_ids[["Latitude", "Longitude"]].drop_duplicates()

glim = gpd.read_file("LiMW_GIS 2015.gdb")
glim = glim.to_crs("EPSG:4326")

stations = gpd.GeoDataFrame(
    station_ids,
    geometry=[Point(lon, lat) for lat, lon in
              zip(station_ids.Latitude, station_ids.Longitude)],
    crs="EPSG:4326"
)

joined = gpd.sjoin(stations, glim, how="left", predicate="within")
print(joined.columns.tolist())   # check column names before saving
joined[["Latitude", "Longitude", "xx", "Litho"]].to_csv("glim_features_TEST.csv", index=False)


