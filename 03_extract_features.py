"""Extract 6-hourly IBTrACS and ERA5 features for the existing ET samples.

Write one CSV per split, with one row per sample and history time. Keep the
input sample order and list each sample's history from the earliest time to
the prediction time. Rows for the same sample share its ID and label. Missing
feature values are written as empty fields.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr

ibtracs = import_module("01_ibtracs_stats")
IBTRACS_FILE = ibtracs.TRACK_FILE
SAMPLE_DIR = ibtracs.OUTPUT_DIR
FEATURE_DIR = SAMPLE_DIR / "features"
ERA5_ROOT = Path("/glade/campaign/collections/gdex/data/d633000")
WIND_DIR = ERA5_ROOT / "e5.oper.an.pl"
SST_DIR = ERA5_ROOT / "e5.oper.an.sfc"
INVARIANT_DIR = ERA5_ROOT / "e5.oper.invariant/197901"
LAND_SEA_MASK = INVARIANT_DIR / "e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc"
LAKE_COVER = INVARIANT_DIR / "e5.oper.invariant.128_026_cl.ll025sc.1979010100_1979010100.nc"
WIND_PARAMETERS = {"U": "131_u", "V": "132_v"}
LEVELS_HPA = (200, 850)
EARTH_RADIUS_KM = 6371.0
RADIUS_KM = 500.0
STEP, HISTORY = ibtracs.STEP, ibtracs.HISTORY
SPLITS = ("train", "val", "test")
# Use the CPU count allocated on Casper. Default to one worker if NCPUS is unset.
WORKERS = int(os.environ.get("NCPUS", "1"))
FEATURE_NAMES = (
    "latitude", "heading_sin", "heading_cos", "central_pressure", "vertical_shear", "sst",
)


def heading_components(latitude, longitude):
    """Return the sine and cosine of the heading measured clockwise from north.

    At each history time, use the segment from the previous position to the
    current position. The first time also uses the first segment in the window,
    so the first two times share the same heading.
    """
    # Arrays have shape (samples, history times). Convert degrees to radians,
    # then pair consecutive positions along the time axis.
    lat = np.deg2rad(latitude)
    lon_change = np.diff(np.deg2rad(longitude), axis=1)
    start_lat, end_lat = lat[:, :-1], lat[:, 1:]

    # Let phi1 and phi2 be the start and end latitudes. Use the start meridian
    # as zero longitude, so the end longitude is delta_lon = lon2 - lon1.
    # On a unit sphere, the endpoint has position vector
    # P = (cos(phi2) * cos(delta_lon), cos(phi2) * sin(delta_lon), sin(phi2)).
    # At the start, the east unit vector is (0, 1, 0) and the north unit vector
    # is (-sin(phi1), 0, cos(phi1)). The dot products of P with these vectors
    # give the east and north components of the initial great-circle direction.
    # E = cos(phi2) * sin(delta_lon)
    # N = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(delta_lon)
    east = np.sin(lon_change) * np.cos(end_lat)
    north = (
        np.cos(start_lat) * np.sin(end_lat)
        - np.sin(start_lat) * np.cos(end_lat) * np.cos(lon_change)
    )

    # The heading is theta = atan2(E, N). Dividing by length = sqrt(E**2 + N**2)
    # gives sin(theta) = E / length and cos(theta) = N / length directly.
    # Moving along the equator from (0, 0) to (0, 10) in (latitude, longitude)
    # degrees gives E = sin(10 degrees) and N = 0. The normalized pair is (1, 0)
    # for due east (90 degrees). Due north (0 degrees) gives (0, 1).
    # The sine and cosine stay continuous as the heading wraps through 360 degrees.
    tangent_norm = np.hypot(east, north)
    # Coincident or antipodal points have a near-zero projection and undefined heading.
    tangent_norm[tangent_norm < 1e-12] = np.nan
    heading = np.stack((east / tangent_norm, north / tangent_norm), axis=-1)
    return np.concatenate((heading[:, :1], heading), axis=1)


@cache
def era5_grid():
    """Read the static ERA5 latitude and longitude grid and identify ocean cells."""
    with xr.open_dataset(LAND_SEA_MASK) as source:
        latitude = source["latitude"].to_numpy()
        longitude = source["longitude"].to_numpy()
        land = source["LSM"].isel(time=0).load()
    with xr.open_dataset(LAKE_COVER) as source:
        land, lake = xr.align(land, source["CL"].isel(time=0), join="exact")
        # Include cells with less than 50% combined land and lake cover in the SST average.
        ocean = (land + lake).to_numpy() < 0.5
    return latitude, longitude, ocean


def area_means(latitude, longitude, u, v, sst):
    """Return 200-850 hPa shear (m/s) and ocean SST (degrees Celsius) within 500 km."""
    grid_lat, grid_lon, ocean = era5_grid()
    center_lat, center_lon = np.deg2rad([latitude, longitude])
    lat = np.deg2rad(grid_lat)
    # Wrap longitude differences to [-pi, pi) to include nearby cells across the date line.
    delta_lon = (np.deg2rad(grid_lon) - center_lon + np.pi) % (2 * np.pi) - np.pi
    # Arc length equals Earth radius times the central angle, so convert 500 km to radians.
    radius = RADIUS_KM / EARTH_RADIUS_KM

    # Narrow the search to a latitude-longitude rectangle. Keep all longitudes if
    # the search circle reaches a pole.
    max_lon = np.pi
    if abs(center_lat) + radius < np.pi / 2:
        max_lon = np.arcsin(np.sin(radius) / np.cos(center_lat))
    rows = np.flatnonzero(np.abs(lat - center_lat) <= radius)
    columns = np.flatnonzero(np.abs(delta_lon) <= max_lon)
    region = np.ix_(rows, columns)
    lat = lat[rows, None]
    # The spherical law of cosines gives the cosine of the central angle d.
    # cos(d) = sin(center_lat) * sin(grid_lat)
    #          + cos(center_lat) * cos(grid_lat) * cos(delta_lon)
    # Cells inside the circle have a smaller d and a larger cos(d) than the
    # boundary. Cell area is proportional to cos(grid_lat) on this regular grid.
    cos_distance = (np.sin(center_lat) * np.sin(lat)
                    + np.cos(center_lat) * np.cos(lat) * np.cos(delta_lon[columns]))
    weights = np.where(cos_distance >= np.cos(radius), np.cos(lat), 0.0)

    # U and V are the eastward and northward wind components. Average both at
    # each level over the same valid cells. The shear in m/s is the length of
    # the difference between the mean wind vectors.
    # shear = sqrt((U200 - U850)**2 + (V200 - V850)**2)
    u, v = u[:, rows[:, None], columns], v[:, rows[:, None], columns]
    wind_cells = (weights > 0) & np.isfinite(u).all(axis=0) & np.isfinite(v).all(axis=0)
    shear = np.nan
    if wind_cells.any():
        u200, u850 = np.average(u[:, wind_cells], axis=1, weights=weights[wind_cells])
        v200, v850 = np.average(v[:, wind_cells], axis=1, weights=weights[wind_cells])
        shear = np.hypot(u200 - u850, v200 - v850)

    sst = sst[region]
    water_cells = (weights > 0) & ocean[region] & np.isfinite(sst)
    mean_sst = np.nan
    if water_cells.any():
        # Convert the mean ERA5 SST from kelvin to degrees Celsius.
        mean_sst = np.average(sst[water_cells], weights=weights[water_cells]) - 273.15
    return shear, mean_sst


def extract_month(storm_times):
    """Read a month of ERA5 data and compute features at the requested storm times."""
    month = storm_times["time"].iloc[0].strftime("%Y%m")
    surface_files = list((SST_DIR / month).glob("e5.oper.an.sfc.128_034_sstk.ll025*.nc"))
    if len(surface_files) != 1:
        raise ValueError(f"Expected one archived SST NetCDF for {month}, found {surface_files}")
    results = []
    with xr.open_dataset(surface_files[0], cache=False) as surface:
        grid_lat, grid_lon, _ = era5_grid()
        if not (np.array_equal(surface["latitude"], grid_lat)
                and np.array_equal(surface["longitude"], grid_lon)):
            raise ValueError(f"SST and land-sea mask grids differ in {surface_files[0]}")
        for day, day_rows in storm_times.groupby(storm_times["time"].dt.strftime("%Y%m%d")):
            paths = [
                WIND_DIR / month / f"e5.oper.an.pl.128_{parameter}.ll025uv.{day}00_{day}23.nc"
                for parameter in WIND_PARAMETERS.values()
            ]
            with xr.open_dataset(paths[0], cache=False) as zonal, xr.open_dataset(
                paths[1], cache=False
            ) as meridional:
                for source, path in zip((zonal, meridional), paths):
                    if not (np.array_equal(source["latitude"], grid_lat)
                            and np.array_equal(source["longitude"], grid_lon)):
                        raise ValueError(f"Wind and land-sea mask grids differ in {path}")
                winds = []
                for source, name in zip((zonal, meridional), WIND_PARAMETERS):
                    if not all(level in source.indexes["level"] for level in LEVELS_HPA):
                        raise ValueError(f"Missing pressure level in {source.encoding['source']}")
                    first, last = source.indexes["level"].get_indexer(LEVELS_HPA)
                    if first > last:
                        raise ValueError(f"Expected increasing pressure levels in {source.encoding['source']}")
                    # Use a strided slice to read both levels in one pass over the GDEX time chunk.
                    winds.append(source[name].isel(level=slice(first, last + 1, last - first)))
                # Read the wind and SST grids once per time and reuse them for all storms at that time.
                for time, centers in day_rows.groupby("time"):
                    u, v = [wind.sel(time=time).to_numpy() for wind in winds]
                    sst = surface["SSTK"].sel(time=time).to_numpy()
                    for center in centers.itertuples():
                        shear, mean_sst = area_means(center.latitude, center.longitude, u, v, sst)
                        results.append((center.Index, shear, mean_sst))
    return np.asarray(results, dtype=float)


def write_feature_csv(path, features, times, metadata):
    """Write a (samples, history times, features) array with one row per sample and time."""
    history_steps = times.shape[1]
    # Flatten features and times in sample order, with all history times for
    # each sample together. Repeat each ID and label to match those rows.
    table = pd.DataFrame({
        "sample_id": np.repeat(metadata["sample_id"].to_numpy(dtype=str), history_steps),
        "history_time": times.ravel(),
        "label": np.repeat(metadata["label"].to_numpy(dtype=np.int64), history_steps),
    })
    table[list(FEATURE_NAMES)] = features.reshape(-1, len(FEATURE_NAMES))

    # Write the complete CSV to a temporary file beside the destination before replacing it.
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        table.to_csv(temporary, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        temporary.replace(path)


if __name__ == "__main__":
    # Load each split's sample list and check that its windows end at the prediction times.
    sample_tables = {}
    for split in SPLITS:
        path = SAMPLE_DIR / f"{split}_samples.csv"
        samples = pd.read_csv(path, keep_default_na=False, dtype={"sid": str, "sample_id": str})
        for field in ("prediction_time", "history_start_time", "history_end_time"):
            samples[field] = pd.to_datetime(samples[field], errors="raise")
        end = samples["prediction_time"]
        sample_tables[split] = samples
    samples = pd.concat(sample_tables, names=["split", "row"])

    # Expand each sample to nine times from 48 hours before prediction through
    # the prediction time itself. Look up each (storm ID, time) pair just once.
    offsets = np.arange(int(HISTORY / STEP) + 1) * STEP.to_timedelta64()
    times = samples["history_start_time"].to_numpy(dtype="datetime64[ns]")[:, None] + offsets
    keys = pd.MultiIndex.from_arrays(
        [np.repeat(samples["sid"].to_numpy(), len(offsets)), times.ravel()], names=["sid", "time"]
    )
    unique_keys = keys.unique()
    # Map each sample and history time to its row in the deduplicated table.
    row_index = unique_keys.get_indexer(keys).reshape(times.shape)

    columns = ["SID", "ISO_TIME", "TRACK_TYPE", "USA_AGENCY", "USA_LAT", "USA_LON", "USA_PRES"]
    reports = pd.read_csv(IBTRACS_FILE, usecols=columns, skiprows=[1], dtype=str, keep_default_na=False)
    reports = reports.loc[
        reports["SID"].isin(samples["sid"])
        & reports["TRACK_TYPE"].eq("main")
        & reports["USA_AGENCY"].str.strip().ne("")
    ].copy()
    reports["ISO_TIME"] = pd.to_datetime(reports["ISO_TIME"], errors="coerce")
    reports = reports.dropna(subset=["ISO_TIME"]).set_index(["SID", "ISO_TIME"])
    centers = reports[["USA_LAT", "USA_LON", "USA_PRES"]].reindex(unique_keys)
    centers = centers.apply(pd.to_numeric, errors="coerce")
    centers.columns = ["latitude", "longitude", "central_pressure"]
    # USA_PRES is in millibars, numerically equal to hPa. Treat nonpositive values as missing.
    centers["central_pressure"] = centers["central_pressure"].where(centers["central_pressure"] > 0)
    if not np.isfinite(centers[["latitude", "longitude"]]).all().all():
        raise ValueError("A sample history is missing a USA position in IBTrACS.")

    # Process ERA5 months in parallel and place each result at its unique storm-time row index.
    era5_requests = centers[["latitude", "longitude"]].reset_index()
    months = [group for _, group in era5_requests.groupby(era5_requests["time"].dt.to_period("M"))]
    environment = np.full((len(centers), 2), np.nan)
    print(f"Extracting {len(centers):,} storm-times from {len(months)} months with {WORKERS} worker(s).",
          flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as workers:
        for count, monthly in enumerate(workers.map(extract_month, months), start=1):
            environment[monthly[:, 0].astype(int)] = monthly[:, 1:]
            if count % 10 == 0 or count == len(months):
                print(f"Processed {count}/{len(months)} ERA5 months.", flush=True)

    # Restore each sample's complete history before computing headings from adjacent positions.
    latitude = centers["latitude"].to_numpy()[row_index]
    longitude = centers["longitude"].to_numpy()[row_index]
    pressure = centers["central_pressure"].to_numpy()[row_index]
    heading = heading_components(latitude, longitude)
    X = np.stack(
        (latitude, heading[:, :, 0], heading[:, :, 1], pressure,
         environment[row_index, 0], environment[row_index, 1]),
        axis=-1,
    ).astype(np.float32)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        split_mask = samples.index.get_level_values("split") == split
        split_X = X[split_mask]
        path = FEATURE_DIR / f"{split}_features.csv"
        write_feature_csv(path, split_X, times[split_mask], sample_tables[split])
        missing = dict(zip(FEATURE_NAMES, np.isnan(split_X).sum(axis=(0, 1)).tolist()))
        sample_count, history_steps, _ = split_X.shape
        print(f"Saved {path} with {sample_count * history_steps:,} rows "
              f"for {sample_count:,} samples. Missing feature values {missing}", flush=True)
