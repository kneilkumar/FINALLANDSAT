"""
Landsat Feature Extraction Pipeline  —  Station-Level Architecture
===================================================================
Performance model
-----------------
  OLD : 9,319 rows  × (Landsat load + index compute + HAS + DEM)  ≈ 181 hrs
  NEW :   162 stations × (Landsat load + index compute + HAS + DEM)  ← expensive, done ONCE
        + 9,319 samples × (in-memory 90d slice + spatial mean)       ← cheap, ~0.1s each

Each Dask worker handles one station end-to-end and returns ALL sample
rows for that station.  No repeated API calls across sample dates.

Spatial tiers
-------------
  t500 –  500m – riparian zone, point sources, bank conditions
  t2k  –  2km  – immediate local land use, small tributary inputs
  t8k  –  8km  – catchment-scale signal (native DEM bbox)

Static features  (computed once per station, broadcast to every sample row)
  Full-period temporal median → spatial mean/std at t500 / t2k / t8k
  Upstream HAS at t2k / t8k

Dynamic features  (per sample date, cheap in-memory slice)
  90-day snapshot z-score anomaly vs station baseline at t500 / t2k / t8k
  NDVI 1st-diff at t500
"""

import datetime
import time

import numpy as np
import pandas as pd
import xarray as xr
from xrspatial import slope

import pystac_client
import planetary_computer
from odc.stac import stac_load

from dask.distributed import Client, LocalCluster
from dask import delayed, compute

import gc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Full observation window — load once per station, slice dates in memory
OBS_START = "2009-10-01"   # buffer before 2010 so 90-day windows work
OBS_END   = "2015-12-31"

SRTM_API_KEY = "3a275809e03b34733e6e20eaaeda247a"

TIER_SIZES = {"t500": 500, "t2k": 2000, "t8k": 8000}

# Which indices to output per tier
TIER_INDICES = {
    "t500": ["NDVI", "NDBI", "BSI"],
    "t2k":  ["NDVI", "NDBI", "BSI", "NDWI", "NDMI"],
    "t8k":  ["NDVI", "NDBI", "NDWI", "NDMI"],
}

# Which indices get z-scored for dynamic features
DYNAMIC_INDICES = {
    "t500": ["NDVI", "BSI"],
    "t2k":  ["NDVI", "NDWI", "BSI", "NDMI"],
    "t8k":  ["NDWI", "NDMI"],
}

SPECTRAL_BANDS = ["red", "green", "blue", "nir08", "swir16", "swir22"]

BIT_FLAGS = {
    "fill": 1 << 0, "dilated_cloud": 1 << 1, "cirrus": 1 << 2,
    "cloud": 1 << 3, "shadow": 1 << 4, "snow": 1 << 5, "water": 1 << 7,
}
INVALID_BITS = (
    BIT_FLAGS["fill"] | BIT_FLAGS["dilated_cloud"] | BIT_FLAGS["cirrus"]
    | BIT_FLAGS["cloud"] | BIT_FLAGS["shadow"] | BIT_FLAGS["snow"]
)

# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def bbox_from_coords(station_coords: tuple, tier_m: int) -> list:
    buf = tier_m / 110_000
    lat, lon = station_coords
    return [lon - buf/2, lat - buf/2, lon + buf/2, lat + buf/2]


def clip_to_tier(ds: xr.Dataset, station_coords: tuple, tier_m: int) -> xr.Dataset:
    bb = bbox_from_coords(station_coords, tier_m)
    return ds.sel(latitude=slice(bb[3], bb[1]), longitude=slice(bb[0], bb[2]))

# ---------------------------------------------------------------------------
# Landsat loading  —  called ONCE per station
# ---------------------------------------------------------------------------

def load_station_stack(station_coords: tuple, catalog) -> xr.Dataset | None:
    """
    Load the full 2010-2015 Landsat stack for a station's 8km bbox.
    Applies L2 scaling + computes all indices, then calls .compute() to
    bring everything into memory.  All subsequent operations are free slices.
    """
    bb = bbox_from_coords(station_coords, 8000)

    items = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bb,
        datetime=f"{OBS_START}/{OBS_END}",
        query={
            "platform":       {"in": ["landsat-7", "landsat-8"]},
            "eo:cloud_cover": {"lt":15},
        },
    ).item_collection()

    if not items:
        return None

    ds = stac_load(
        items,
        bands=SPECTRAL_BANDS + ["qa_pixel"],
        crs="EPSG:4326",
        resolution=30 / 111_320.0,
        chunks={"time": 1, "x": 512, "y": 512},
        patch_url=planetary_computer.sign,
        bbox=bb,
    )

    for band in SPECTRAL_BANDS:
        ds[band] = ds[band] * 0.0000275 - 0.2

    ds = _add_indices(ds)

    print(f"  [{station_coords}] computing full stack ({ds.sizes['time']} scenes)…")
    ds = ds.compute()
    print(f"  [{station_coords}] stack in memory.")
    return ds


def _add_indices(ds: xr.Dataset) -> xr.Dataset:
    eps = 1e-10
    ds["NDVI"]  = (ds["nir08"] - ds["red"])    / (ds["nir08"] + ds["red"]    + eps)
    ds["NDWI"]  = (ds["green"] - ds["nir08"])  / (ds["green"] + ds["nir08"] + eps)
    ds["NDBI"]  = (ds["swir16"] - ds["nir08"]) / (ds["swir16"] + ds["nir08"] + eps)
    ds["NDMI"]  = (ds["nir08"] - ds["swir16"]) / (ds["nir08"] + ds["swir16"] + eps)
    ds["BSI"]   = (
        (ds["swir16"] + ds["red"]) - (ds["nir08"] + ds["blue"])
    ) / (ds["swir16"] + ds["red"] + ds["nir08"] + ds["blue"] + eps)
    return ds


def _cloud_mask(qa: xr.DataArray) -> xr.DataArray:
    return (qa & INVALID_BITS) != 0

def _water_mask(qa: xr.DataArray) -> xr.DataArray:
    return (qa & BIT_FLAGS["water"]) != 0

# ---------------------------------------------------------------------------
# Annual baseline  —  one dict per year per station
# ---------------------------------------------------------------------------

def compute_annual_baselines(ds: xr.Dataset, station_coords: tuple) -> dict:
    """
    Compute one baseline per calendar year present in the stack.
    A sample in 2011 gets the 2011 land state, not a 2010-2015 average.

    Returns:
        { year (int) -> { "t500_static_NDVI": float, ... } }

    Std entries are used for z-score normalisation of dynamic features.
    HAS stays truly static — land cover and DEM don't change meaningfully
    year-to-year at the scale we care about.
    """
    bad = _cloud_mask(ds["qa_pixel"])
    wet = _water_mask(ds["qa_pixel"])

    all_indices = ["NDVI", "NDWI", "NDBI", "NDMI", "BSI"]
    clean = ds[all_indices].where(~bad)
    clean["NDVI"] = clean["NDVI"].where(~wet)
    clean["NDMI"] = clean["NDMI"].where(~wet)

    years = sorted(set(pd.DatetimeIndex(ds.time.values).year))
    annual = {}

    for year in years:
        year_ds = clean.sel(time=str(year))
        if year_ds.sizes["time"] == 0:
            continue

        baseline = {}
        for tier_name, tier_m in TIER_SIZES.items():
            tier_ds = clip_to_tier(year_ds, station_coords, tier_m)
            indices = TIER_INDICES[tier_name]
            dyn_idx = DYNAMIC_INDICES[tier_name]

            med = tier_ds[indices].median(dim="time", skipna=True)
            for idx in indices:
                baseline[f"{tier_name}_static_{idx}"] = float(
                    med[idx].mean(dim=["latitude", "longitude"], skipna=True)
                )

            std = tier_ds[dyn_idx].std(dim="time", skipna=True)
            for idx in dyn_idx:
                baseline[f"{tier_name}_std_{idx}"] = float(
                    std[idx].mean(dim=["latitude", "longitude"], skipna=True)
                )

        annual[year] = baseline

    # Fill any missing years by borrowing nearest available
    all_years = list(range(min(years), max(years) + 1))
    for year in all_years:
        if year not in annual:
            candidates = sorted(annual.keys(), key=lambda y: abs(y - year))
            annual[year] = annual[candidates[0]]
            print(f"  [{station_coords}] no imagery for {year}, borrowing from {candidates[0]}")

    return annual

# ---------------------------------------------------------------------------
# HAS  —  computed ONCE per station
# ---------------------------------------------------------------------------

def compute_station_has(ds: xr.Dataset, station_coords: tuple) -> dict:
    """
    Upstream Human Activity Signatures using full-period Landsat mean + DEM.
    Same values broadcast to all sample rows for this station.
    """
    from bmi_topography import Topography

    bb = bbox_from_coords(station_coords, 8000)
    params = Topography.DEFAULT.copy()
    params.update({
        "west": bb[0], "south": bb[1], "east": bb[2], "north": bb[3],
        "dem_type": "SRTMGL1",
    })
    dem_obj = Topography(**params)
    dem_obj.fetch()
    dem = dem_obj.load().squeeze().rename({"x": "longitude", "y": "latitude"}) / 1000.0

    has_features = {}
    for tier_name, tier_m in [("t2k", 2000), ("t8k", 8000)]:
        tier_ds = clip_to_tier(ds, station_coords, tier_m)

        dem_tier = dem.sel(
            latitude=tier_ds.latitude,
            longitude=tier_ds.longitude,
            method="nearest",
        )
        station_elev = float(
            dem.sel(
                latitude=station_coords[0],
                longitude=station_coords[1],
                method="nearest",
            )
        )

        rel_elev = np.maximum(0, dem_tier - station_elev)
        rel_elev = rel_elev.where(rel_elev > 0)

        lat_km  = (dem_tier["latitude"]  - station_coords[0]) * 111.0
        lon_km  = (dem_tier["longitude"] - station_coords[1]) * 111.0 * np.cos(
            np.pi * station_coords[0] / 180.0
        )
        dist_km = np.sqrt(lat_km**2 + lon_km**2)
        terr_slope = slope(dem_tier)

        w_ec  = rel_elev * terr_slope / np.maximum(dist_km,       0.03)
        w_drp = rel_elev * terr_slope / np.maximum(dist_km**1.5,  0.03**1.5)

        bad      = _cloud_mask(tier_ds["qa_pixel"])
        wet      = _water_mask(tier_ds["qa_pixel"])
        lc       = tier_ds[["NDVI", "NDBI", "BSI"]].where(~bad)
        lc["NDVI"] = lc["NDVI"].where(~wet)
        ndvi_std = lc["NDVI"].std(dim="time", skipna=True)
        bsi_std  = lc["BSI"].std(dim="time",  skipna=True)
        mean_img = lc.mean(dim="time", skipna=True)

        urban = (
            (mean_img["NDBI"] > 0.013) & (mean_img["NDVI"] < 0.10)
            & (ndvi_std < 0.10)        & (mean_img["NDVI"] > 0.02)
        )
        mine = (
            (mean_img["BSI"] > 0.10) & (mean_img["NDVI"] < 0.15)
            & (ndvi_std < 0.10)      & (mean_img["NDVI"] > 0.02)
        )
        farm = (ndvi_std > 0.10) & (bsi_std > 0.10) & (mean_img["NDVI"] > 0.24)

        w_ec  = w_ec.assign_coords(latitude=mean_img.latitude,  longitude=mean_img.longitude)
        w_drp = w_drp.assign_coords(latitude=mean_img.latitude, longitude=mean_img.longitude)
        tot_ec  = float(w_ec.sum(skipna=True))  + 1e-10
        tot_drp = float(w_drp.sum(skipna=True)) + 1e-10

        def wsig(w, da, mask, total):
            return float((w * da.where(mask)).sum(skipna=True) / total)

        has_features.update({
            f"{tier_name}_has_urban_sig_ec":  wsig(w_ec,  mean_img["NDBI"], urban, tot_ec),
            f"{tier_name}_has_urban_sig_drp": wsig(w_drp, mean_img["NDBI"], urban, tot_drp),
            f"{tier_name}_has_mine_sig_ec":   wsig(w_ec,  mean_img["BSI"],  mine,  tot_ec),
            f"{tier_name}_has_mine_sig_drp":  wsig(w_drp, mean_img["BSI"],  mine,  tot_drp),
            f"{tier_name}_has_farm_sig_ec":   wsig(w_ec,  mean_img["NDVI"], farm,  tot_ec),
            f"{tier_name}_has_farm_sig_drp":  wsig(w_drp, mean_img["NDVI"], farm,  tot_drp),
        })

    return has_features

# ---------------------------------------------------------------------------
# Dynamic features  —  cheap per-sample-date (all in-memory)
# ---------------------------------------------------------------------------

def _best_snapshot(tier_ds: xr.Dataset, sample_date: pd.Timestamp) -> xr.Dataset | None:
    """
    Find the cleanest single scene closest to sample_date within tier_ds.
    Falls back to temporal median composite if no scene is >80% valid.
    """
    if tier_ds.sizes["time"] == 0:
        return None

    bad = _cloud_mask(tier_ds["qa_pixel"])
    wet = _water_mask(tier_ds["qa_pixel"])

    valid_ratio = (~bad).sum(dim=["latitude","longitude"]) / (
        bad.sizes["latitude"] * bad.sizes["longitude"]
    )
    clean_times = tier_ds["time"].where(valid_ratio > 0.8).dropna(dim="time")

    indices     = ["NDVI", "NDWI", "NDBI", "NDMI", "BSI"]
    clean_stack = tier_ds[indices].where(~bad)
    clean_stack["NDVI"] = clean_stack["NDVI"].where(~wet)
    clean_stack["NDMI"] = clean_stack["NDMI"].where(~wet)

    if clean_times.sizes["time"] > 0:
        deltas  = np.abs(clean_times.values - np.datetime64(sample_date))
        closest = clean_times.isel(time=int(np.argmin(deltas)))
        return clean_stack.sel(time=closest)
    else:
        return clean_stack.median(dim="time", skipna=True)


def extract_dynamic_features(
    ds: xr.Dataset,
    station_coords: tuple,
    sample_date: pd.Timestamp,
    annual_baselines: dict,
) -> dict:
    """
    Slice the in-memory station stack to a 90-day window and compute
    z-score anomalies against that sample year's baseline.  No I/O.
    """
    # Use the baseline for the sample's year; fall back to nearest if missing
    year = sample_date.year
    if year not in annual_baselines:
        year = min(annual_baselines.keys(), key=lambda y: abs(y - year))
    baseline = annual_baselines[year]

    window_start = sample_date - datetime.timedelta(days=90)
    ds_90 = ds.sel(time=slice(window_start, sample_date))

    if ds_90.sizes["time"] == 0:
        return {}

    features = {}
    for tier_name, tier_m in TIER_SIZES.items():
        tier_90 = clip_to_tier(ds_90, station_coords, tier_m)
        snap    = _best_snapshot(tier_90, sample_date)
        if snap is None:
            continue

        snap_mean = snap.mean(dim=["latitude", "longitude"], skipna=True)

        for idx in DYNAMIC_INDICES[tier_name]:
            snap_val = float(snap_mean[idx]) if idx in snap_mean else np.nan
            base_val = baseline.get(f"{tier_name}_static_{idx}", np.nan)
            base_std = baseline.get(f"{tier_name}_std_{idx}",    np.nan)
            features[f"{tier_name}_90d_{idx}_anomaly"] = (
                (snap_val - base_val) / (base_std + 1e-10)
            )

        if tier_name == "t500":
            features["t500_90d_NDVI_1st_diff"] = (
                float(snap_mean["NDVI"]) - baseline.get("t500_static_NDVI", np.nan)
            )

    return features

# ---------------------------------------------------------------------------
# Station worker  —  the Dask unit of parallelism
# ---------------------------------------------------------------------------

@delayed
def process_station(
    station_coords: tuple,
    sample_dates: list,
    catalog,
) -> list:
    """
    Process one station completely:
      1. Load full Landsat stack once          (~60-150s, one-time cost)
      2. Compute annual baselines (6 years)    (~5s total)
      3. Compute HAS once                      (~10s)
      4. Loop over sample dates cheaply        (in-memory per date)

    Each sample row gets the baseline for its own calendar year, so 2011
    and 2015 samples will have different static feature values.

    Returns a list of dicts (one per sample date) ready for pd.DataFrame.
    """
    lat, lon = station_coords
    tag = f"[{lat:.4f}, {lon:.4f}]"
    print(f"{tag} starting — {len(sample_dates)} samples")

    ds = load_station_stack(station_coords, catalog)
    if ds is None:
        print(f"{tag} no imagery — returning NaN rows")
        return [{"Latitude": lat, "Longitude": lon, "Sample Date": d}
                for d in sample_dates]

    # Annual baselines: { year -> { feature_name -> float } }
    annual_baselines = compute_annual_baselines(ds, station_coords)

    has_features = {}
    try:
        has_features = compute_station_has(ds, station_coords)
    except Exception as e:
        print(f"{tag} HAS failed: {e}")

    rows = []
    for sample_date in sample_dates:
        # Get this sample's year baseline for static features
        year = sample_date.year
        if year not in annual_baselines:
            year = min(annual_baselines.keys(), key=lambda y: abs(y - year))
        year_baseline = annual_baselines[year]

        dynamic = extract_dynamic_features(ds, station_coords, sample_date, annual_baselines)

        rows.append({
            "Latitude":    lat,
            "Longitude":   lon,
            "Sample Date": sample_date,
            **year_baseline,   # static features for this sample's year
            **has_features,    # HAS is truly static — DEM/land cover don't shift year-to-year
            **dynamic,
        })

    del ds

    gc.collect()

    print(f"{tag} done — {len(rows)} rows produced")
    return rows

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    train_set = pd.read_csv("water_quality_training_dataset.csv")
    train_set["Sample Date dt"] = pd.to_datetime(train_set["Sample Date"], dayfirst=True)
    train_set["Latitude"]  = train_set["Latitude"].round(6)
    train_set["Longitude"] = train_set["Longitude"].round(6)

    # Group by station — one task per station, not per row
    station_groups = (
        train_set
        .groupby(["Latitude", "Longitude"])["Sample Date dt"]
        .apply(list)
        .reset_index()
    )
    print(f"Stations to process: {len(station_groups)}")

    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client  = Client(cluster)
    print(client.dashboard_link)

    import os

    OUTPUT_DIR = "ls_station_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BATCH_SIZE = 1
    all_sids = list(range(130,162))
    batches = [all_sids[i:i + BATCH_SIZE] for i in range(0, len(all_sids), BATCH_SIZE)]

    for batch_num, batch in enumerate(batches):
        pending = [
            sid for sid in batch
            if not os.path.exists(f"{OUTPUT_DIR}/ls_station_{sid}.csv")
        ]

        if not pending:
            print(f"Batch {batch_num + 1}/{len(batches)} — all done, skipping.")
            continue

        print(f"\nBatch {batch_num + 1}/{len(batches)} — processing {len(pending)} stations…")

        tasks = {}
        for sid in pending:
            row = station_groups.iloc[sid]
            tasks[sid] = process_station(
                (row["Latitude"], row["Longitude"]),
                row["Sample Date dt"],
                catalog,
            )

        t0 = time.perf_counter()
        try:
            sids_list = list(tasks.keys())
            results_list = compute(*tasks.values())

            for sid, result in zip(sids_list, results_list):
                output_path = f"{OUTPUT_DIR}/ls_station_{sid}.csv"
                pd.DataFrame(result).to_csv(output_path, index=False)
                print(f"  station {sid} saved → {output_path}")

        except Exception as e:
            print(f"  Batch failed ({e}) — retrying individually…")
            for sid in pending:
                output_path = f"{OUTPUT_DIR}/ls_station_{sid}.csv"
                if os.path.exists(output_path):
                    continue
                row = station_groups.iloc[sid]
                try:
                    result = compute(process_station(
                        (row["Latitude"], row["Longitude"]),
                        row["Sample Date dt"],
                        catalog,
                    ))[0]
                    pd.DataFrame(result).to_csv(output_path, index=False)
                    print(f"  station {sid} saved (retry)")
                except Exception as e2:
                    print(f"  station {sid} FAILED: {e2}")

        print(f"Batch {batch_num + 1} done in {time.perf_counter() - t0:.0f}s")
