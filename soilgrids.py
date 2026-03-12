"""
SoilGrids GeoTIFF Feature Extraction
======================================
Reads local SoilGrids GeoTIFFs and extracts features for each station at
500m and 8km spatial tiers, plus an upstream/downstream split using the
SRTM DEM elevation proxy.

Expected input files (place in TIFF_DIR):
  clay_5-15cm.tif
  phh2o_5-15cm.tif
  cec_5-15cm.tif
  soc_5-15cm.tif

Output: soilgrids_features.csv  (one row per station)
"""

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import rasterio.transform
from rasterio.warp import transform as rio_transform
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIFF_DIR = Path("soilgrids_tiffs")

# filename stem -> feature prefix
TIFF_FILES = {
    "clay_5-15cm":  "clay",
    "phh2o_5-15cm": "ph",
    "cec_5-15cm":   "cec",
    "soc_5-15cm":   "soc",
}

TIERS = {"t500": 500, "t8k": 8000}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bbox_from_coords(lat: float, lon: float, tier_m: int) -> tuple:
    buf = tier_m / 110_000
    return lon - buf/2, lat - buf/2, lon + buf/2, lat + buf/2


def sample_raster_in_bbox(
    src: rasterio.io.DatasetReader,
    lat: float,
    lon: float,
    tier_m: int,
) -> np.ndarray:
    """
    Sample all raster pixels within the tier bounding box.
    Handles both EPSG:4326 and projected CRS automatically.
    Returns a flat array of valid (non-nodata) values.
    """
    west, south, east, north = bbox_from_coords(lat, lon, tier_m)

    if not src.crs.is_geographic:
        xs, ys  = rio_transform("EPSG:4326", src.crs, [west, east], [south, north])
        west, east = xs[0], xs[1]
        south, north = ys[0], ys[1]

    window = rasterio.windows.from_bounds(
        west, south, east, north,
        transform=src.transform,
    )
    data = src.read(1, window=window).astype(float)

    if src.nodata is not None:
        data[data == src.nodata] = np.nan

    return data[~np.isnan(data)]


def get_dem_elevations(lat: float, lon: float):
    """
    Fetch SRTM DEM for the 8km bbox via bmi_topography (cached after first call).
    Returns (dem_xarray, station_elevation_m).
    """
    from bmi_topography import Topography

    west, south, east, north = bbox_from_coords(lat, lon, 8000)
    params = Topography.DEFAULT.copy()
    params.update({
        "west": west, "south": south, "east": east, "north": north,
        "dem_type": "SRTMGL1",
    })
    dem_obj = Topography(**params)
    dem_obj.fetch()
    dem = dem_obj.load().squeeze().rename({"x": "longitude", "y": "latitude"})
    station_elev = float(dem.sel(latitude=lat, longitude=lon, method="nearest"))
    return dem, station_elev


def sample_upstream_downstream(
    src: rasterio.io.DatasetReader,
    lat: float,
    lon: float,
    dem,
    station_elev: float,
) -> tuple:
    """
    Split 8km bbox pixels into upstream (elev > station) and downstream.
    Returns (upstream_values, downstream_values) as numpy arrays.
    """
    west, south, east, north = bbox_from_coords(lat, lon, 8000)

    proj_west, proj_east = west, east
    proj_south, proj_north = south, north
    if not src.crs.is_geographic:
        xs, ys = rio_transform("EPSG:4326", src.crs, [west, east], [south, north])
        proj_west, proj_east = xs[0], xs[1]
        proj_south, proj_north = ys[0], ys[1]

    window     = rasterio.windows.from_bounds(
        proj_west, proj_south, proj_east, proj_north,
        transform=src.transform,
    )
    data       = src.read(1, window=window).astype(float)
    if src.nodata is not None:
        data[data == src.nodata] = np.nan

    win_tf     = rasterio.windows.transform(window, src.transform)
    rows, cols = np.where(~np.isnan(data))
    pxs, pys   = rasterio.transform.xy(win_tf, rows, cols)

    if not src.crs.is_geographic:
        pxs, pys = rio_transform(src.crs, "EPSG:4326", pxs, pys)

    upstream, downstream = [], []
    for i, (plat, plon) in enumerate(zip(pys, pxs)):
        elev = float(dem.sel(latitude=plat, longitude=plon, method="nearest"))
        val  = data[rows[i], cols[i]]
        if elev > station_elev:
            upstream.append(val)
        else:
            downstream.append(val)

    return np.array(upstream), np.array(downstream)


# ---------------------------------------------------------------------------
# Per-station extraction
# ---------------------------------------------------------------------------

def extract_station(
    station_lat: float,
    station_lon: float,
    open_rasters: dict,
) -> dict:
    features = {"Latitude": station_lat, "Longitude": station_lon}

    # DEM — fetched once per station
    try:
        dem, station_elev = get_dem_elevations(station_lat, station_lon)
        has_dem = True
    except Exception as e:
        print(f"  DEM failed: {e}")
        has_dem = False

    for file_stem, feat_prefix in TIFF_FILES.items():
        src = open_rasters.get(file_stem)
        if src is None:
            continue

        # Spatial means + std at each tier
        for tier_name, tier_m in TIERS.items():
            vals = sample_raster_in_bbox(src, station_lat, station_lon, tier_m)
            features[f"sg_{feat_prefix}_{tier_name}_mean"] = (
                float(np.mean(vals)) if len(vals) > 0 else np.nan
            )
            features[f"sg_{feat_prefix}_{tier_name}_std"] = (
                float(np.std(vals))  if len(vals) > 0 else np.nan
            )

        # Upstream / downstream split
        if has_dem:
            try:
                up, down  = sample_upstream_downstream(
                    src, station_lat, station_lon, dem, station_elev
                )
                up_mean   = float(np.mean(up))   if len(up)   > 0 else np.nan
                down_mean = float(np.mean(down)) if len(down) > 0 else np.nan
                features[f"sg_{feat_prefix}_upstream_mean"]   = up_mean
                features[f"sg_{feat_prefix}_downstream_mean"] = down_mean
                features[f"sg_{feat_prefix}_updown_ratio"]    = (
                    up_mean / (down_mean + 1e-10)
                    if not (np.isnan(up_mean) or np.isnan(down_mean))
                    else np.nan
                )
            except Exception as e:
                print(f"  Upstream split failed for {feat_prefix}: {e}")

    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Warn about any missing TIFFs
    missing = [f for f in TIFF_FILES if not (TIFF_DIR / f"{f}.tif").exists()]
    if missing:
        print(f"WARNING — missing TIFFs: {missing}")

    train_set = pd.read_csv("submission_template.csv")
    train_set["Latitude"]  = train_set["Latitude"].round(6)
    train_set["Longitude"] = train_set["Longitude"].round(6)

    stations = (
        train_set[["Latitude", "Longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"Processing {len(stations)} stations…")

    # Open all rasters once and keep handles open across stations
    open_rasters = {}
    for file_stem in TIFF_FILES:
        path = TIFF_DIR / f"{file_stem}.tif"
        if path.exists():
            open_rasters[file_stem] = rasterio.open(path)
            print(f"  Opened {path.name} | CRS: {open_rasters[file_stem].crs} | "
                  f"Bounds: {open_rasters[file_stem].bounds}")
        else:
            print(f"  Not found: {path}")

    OUTPUT  = "soilgrids_features_TEST.csv"
    results = []

    for i, row in stations.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        print(f"\n[{i+1}/{len(stations)}] ({lat}, {lon})")
        result = extract_station(lat, lon, open_rasters)
        results.append(result)
        pd.DataFrame(results).to_csv(OUTPUT, index=False)   # incremental save

    for src in open_rasters.values():
        src.close()

    final = pd.DataFrame(results)
    print(f"\nDone. Shape: {final.shape} | Saved to {OUTPUT}")
    print(final.head())