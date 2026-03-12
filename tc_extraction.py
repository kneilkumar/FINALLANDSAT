import warnings
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster
from dask import delayed, compute

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NETCDF_DIR = Path("terraclimate_data")

VARIABLES = [
    "q",    # runoff
    "def",  # deficit
    "ppt",  # precipitation
    "tmax", # max temperature
    "soil", # soil moisture
    "PDSI", # Palmer Drought Severity Index
]

SA_LAT     = slice(-21.72, -35.18)
SA_LON     = slice(14.97, 32.79)
TIME_START = "2009-10-01"   # buffer before 2010 so 12m rolling works
TIME_END   = "2015-12-31"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_local_terraclimate() -> xr.Dataset:
    datasets = []
    for var in VARIABLES:
        files = sorted(NETCDF_DIR.glob(f"TerraClimate_{var}_*.nc"))
        if not files:
            raise FileNotFoundError(
                f"No files found for '{var}' in {NETCDF_DIR}. "
                f"Expected: TerraClimate_{var}_<year>.nc"
            )
        print(f"  Loading '{var}' from {len(files)} file(s)")
        datasets.append(xr.open_mfdataset(files, combine="by_coords"))

    ds = xr.merge(datasets)
    ds = ds.sel(time=slice(TIME_START, TIME_END), lat=SA_LAT, lon=SA_LON)
    print(f"Loaded: {ds}")
    return ds


# ---------------------------------------------------------------------------
# Per-station feature extraction
# ---------------------------------------------------------------------------

@delayed
def extract_station(
    ds: xr.Dataset,
    lat_i: int,
    lon_i: int,
    station_lat: float,
    station_lon: float,
    sample_dates: list,
) -> list:
    """
    For one station:
      1. Extract 3x3 neighbourhood once (tiny spatial footprint)
      2. Spatially average to a 1D time series per variable
      3. Compute rolling stats on that tiny time series (fast)
      4. For each sample date, look up the prior-month snapshot

    Derived features (z_score, seasonal_contrast, first_diff, second_diff)
    are intentionally omitted — they are trivially computable from
    mean/std/lag columns during feature assembly and don't need to be
    computed over the full spatial grid.

    Returns a list of dicts, one per sample date.
    """
    # Step 1: 3x3 neighbourhood → spatial mean → 1D time series
    lat_start = max(0, lat_i - 1)
    lat_end   = min(lat_i + 2, ds.sizes["lat"])
    lon_start = max(0, lon_i - 1)
    lon_end   = min(lon_i + 2, ds.sizes["lon"])

    aoi = ds.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end))
    ts  = aoi.mean(dim=["lat", "lon"], skipna=True)

    # Step 2: rolling stats on the 1D time series
    mean_3m  = ts.rolling(time=3,  center=False).mean()
    mean_12m = ts.rolling(time=12, center=False).mean()
    std_3m   = ts.rolling(time=3,  center=False).std()
    std_12m  = ts.rolling(time=12, center=False).std()
    lag1     = ts.shift(time=1)
    lag2     = ts.shift(time=2)
    lag3     = ts.shift(time=3)

    def _rename(d, prefix):
        return d.rename({v: f"{prefix}{v}" for v in d.data_vars})

    feat_ts = xr.merge([
        _rename(mean_3m,  "3m_mean_"),
        _rename(mean_12m, "12m_mean_"),
        _rename(std_3m,   "3m_std_"),
        _rename(std_12m,  "12m_std_"),
        _rename(lag1,     "1m_lag_"),
        _rename(lag2,     "2m_lag_"),
        _rename(lag3,     "3m_lag_"),
    ])

    # Step 3: look up prior-month value per sample date
    rows = []
    for sample_date in sample_dates:
        prior_month = sample_date - datetime.timedelta(days=30)
        snap        = feat_ts.sel(time=prior_month, method="nearest")
        row         = snap.to_array().to_series().to_dict()
        row["Sample Date"] = sample_date
        row["Latitude"]    = station_lat
        row["Longitude"]   = station_lon
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trainset = pd.read_csv("submission_template.csv")
    trainset["Sample Date dt"] = pd.to_datetime(trainset["Sample Date"], dayfirst=True)

    station_ids = trainset[["Latitude", "Longitude"]].drop_duplicates()
    date_arrays = (
        trainset
        .groupby(["Latitude", "Longitude"], as_index=False)["Sample Date dt"]
        .apply(list)
        .reset_index()
    )

    print("Loading TerraClimate…")
    ds = load_local_terraclimate()
    print("TerraClimate loaded.\n")

    # Match stations to nearest grid cells
    grid_lat     = ds.lat.values
    grid_lon     = ds.lon.values
    station_lats = station_ids["Latitude"].to_numpy()
    station_lons = station_ids["Longitude"].to_numpy()

    lat_indices = [np.argmin(np.abs(grid_lat - lat)) for lat in station_lats]
    lon_indices = [np.argmin(np.abs(grid_lon - lon)) for lon in station_lons]

    tc_matched = pd.concat([
        pd.DataFrame({"lat_index": lat_indices, "lon_index": lon_indices}),
        date_arrays,
    ], axis=1)

    # One delayed task per station — not per sample row
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client  = Client(cluster)

    tasks = []
    for i, row in tc_matched.iterrows():
        task = extract_station(
            ds,
            row["lat_index"],
            row["lon_index"],
            row["Latitude"],
            row["Longitude"],
            row["Sample Date dt"],
        )
        tasks.append(task)

    print(f"Running {len(tasks)} station tasks…")
    results = compute(*tasks)

    # Flatten list of lists → DataFrame
    all_rows = [row for station_rows in results for row in station_rows]
    output   = pd.DataFrame(all_rows)
    output.to_csv("tc_features_TEST.csv", index=False)
    print(f"Done. {output.shape} written to tc_features.csv")

    client.close()
    cluster.close()