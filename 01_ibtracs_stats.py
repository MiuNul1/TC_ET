"""Count tropical cyclones and reliable ET events in IBTrACS."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT = Path(__file__).resolve().parent
TRACK_FILE = PROJECT / "data/raw/Ibtracks/ibtracs.ALL.list.v04r01.csv"
OUTPUT_DIR = PROJECT / "data/processed"
BASINS = ("WP", "EP", "NA")
FIRST_SEASON, LAST_SEASON = 1980, 2025
STEP = pd.Timedelta(hours=6)  # Spacing between forecast times and between input observations.
HISTORY = pd.Timedelta(hours=48)  # Look back 48 h from the forecast time, including that time.
HORIZON = pd.Timedelta(hours=48)  # Look ahead 48 h from the forecast time to assign the label.
MAX_REPORT_GAP = pd.Timedelta(hours=6)  # Largest acceptable gap between reports used for labels.
TROPICAL_STATUSES = ("TD", "TS", "HU", "HR", "TY", "ST", "TC")
NON_ET_STATUSES = TROPICAL_STATUSES + ("DB", "SD", "SS")
EXTRATROPICAL_STATUS = "EX"


def read_tracks():
    """Read IBTrACS and return USA reports for eligible main tracks.

    The returned pandas DataFrame has one row per retained report, sorted by
    storm ID (SID) and time. A storm qualifies when its first tropical report
    falls in a target basin. Its subsequent reports can establish ET.
    """
    # Keep "NA" as a basin code and skip the CSV's units row.
    fields = ["SID", "SEASON", "BASIN", "ISO_TIME", "TRACK_TYPE",
              "USA_AGENCY", "USA_STATUS", "USA_LAT", "USA_LON"]
    tracks = pd.read_csv(
        TRACK_FILE, usecols=fields, skiprows=[1], dtype=str, keep_default_na=False
    )
    tracks["SEASON"] = pd.to_numeric(tracks["SEASON"], errors="coerce")
    in_range = tracks["SEASON"].between(FIRST_SEASON, LAST_SEASON)
    # A missing agency name marks a gap. Keep reports between grid times for ET timing.
    has_usa_report = tracks["USA_AGENCY"].str.strip().ne("")
    tracks = tracks.loc[in_range & tracks["TRACK_TYPE"].eq("main") & has_usa_report].copy()
    tracks["ISO_TIME"] = pd.to_datetime(
        tracks["ISO_TIME"], format="%Y-%m-%d %H:%M:%S", errors="coerce",
    )
    tracks = tracks.dropna(subset=["ISO_TIME"]).sort_values(["SID", "ISO_TIME"])
    tracks["USA_STATUS"] = tracks["USA_STATUS"].str.strip()

    # Keep the whole USA track for storms that reach tropical cyclone status.
    genesis = tracks.loc[tracks["USA_STATUS"].isin(TROPICAL_STATUSES)].drop_duplicates("SID")
    sids = genesis.loc[genesis["BASIN"].isin(BASINS), "SID"]
    tracks = tracks.loc[tracks["SID"].isin(sids)].copy()
    tracks["latitude"] = pd.to_numeric(tracks["USA_LAT"], errors="coerce")
    longitude = pd.to_numeric(tracks["USA_LON"], errors="coerce")
    tracks["longitude"] = (longitude + 180) % 360 - 180
    return tracks.drop(columns=["TRACK_TYPE", "USA_AGENCY", "USA_LAT", "USA_LON"])


def position_histories(track, predictions):
    """Look up the 48-hour input positions for each candidate forecast time.

    `track` is the DataFrame of USA reports for one storm. `predictions` is a
    pandas DatetimeIndex of candidate forecast times in UTC. Each forecast
    time is the final timestamp in its own input window.

    Return an xarray Dataset with one row per forecast time and nine
    position slots per row. Missing reports leave NaN coordinates.
    """
    # Build the offsets from -48 h to 0 h at 6 h intervals.
    offsets = np.arange(-int(HISTORY / STEP), 1) * STEP.to_timedelta64()
    # Put forecast times in one column so NumPy adds every offset to each time.
    cutoffs = predictions.to_numpy(dtype="datetime64[ns]")[:, None]
    times = cutoffs + offsets
    positions = track.set_index("ISO_TIME")[["latitude", "longitude"]]
    # Look up the flattened times in order, then reshape the coordinates into windows.
    positions = positions.reindex(times.ravel())
    dims = ("prediction_time", "history_step")
    return xr.Dataset(
        {
            "latitude": (dims, positions["latitude"].to_numpy().reshape(times.shape)),
            "longitude": (dims, positions["longitude"].to_numpy().reshape(times.shape)),
        },
        coords={"prediction_time": predictions, "history_step": offsets,
                "history_time": (dims, times)},
    )


def tropical_cutoffs(reports, predictions):
    """Check whether the storm is tropical at each candidate forecast time.

    `reports` is a DataFrame for one storm. `predictions` holds candidate
    forecast timestamps. Return one boolean per timestamp in the same order.
    A missing report gives False.
    """
    statuses = reports.set_index("ISO_TIME")["USA_STATUS"].reindex(predictions)
    return statuses.isin(TROPICAL_STATUSES).to_numpy()


def storm_catalog(tracks):
    """Summarize ET and history eligibility for every storm in `tracks`.

    `tracks` contains the USA reports returned by read_tracks(). Return a
    DataFrame indexed by SID with the first tropical time, the confirmed ET
    time (NaT if unavailable), and a flag for usable pre-ET history.
    """
    storms = []
    for sid, track in tracks.groupby("SID", sort=False):
        reports = track.reset_index(drop=True)
        genesis = reports.loc[reports["USA_STATUS"].isin(TROPICAL_STATUSES)].iloc[0]
        tropical_start = genesis["ISO_TIME"]
        after_genesis = reports["ISO_TIME"].gt(tropical_start)
        et_rows = reports.index[after_genesis & reports["USA_STATUS"].eq(EXTRATROPICAL_STATUS)]
        et_time = pd.NaT
        if not et_rows.empty:
            # The first EX must directly follow a tropical status within six hours.
            first_et = reports.loc[et_rows[0], "ISO_TIME"]
            previous = reports.loc[et_rows[0] - 1]
            if (previous["USA_STATUS"] in TROPICAL_STATUSES
                    and first_et - previous["ISO_TIME"] <= MAX_REPORT_GAP):
                et_time = first_et

        usable = False
        if et_time - tropical_start >= HISTORY + HORIZON:
            # The earliest positive cutoff can be 48 h before ET and needs 48 h of input.
            # Round down to check a position at or before that 96 h boundary.
            history_start = (et_time - HISTORY - HORIZON).floor(STEP)
            last_cutoff = et_time.ceil(STEP) - STEP
            required_times = pd.date_range(history_start, last_cutoff, freq=STEP)
            position_times = track.dropna(subset=["latitude", "longitude"])["ISO_TIME"]
            complete = required_times.isin(position_times).all()
            # Only the cutoffs in the 48 h prediction horizon need a tropical status.
            positive_times = required_times[required_times >= et_time - HORIZON]
            usable = complete and tropical_cutoffs(reports, positive_times).all()
        storms.append({
            "sid": sid,
            "season": int(genesis["SEASON"]),
            "basin": genesis["BASIN"],
            "tropical_start": tropical_start,
            "et_time": et_time,
            "has_et_report": not et_rows.empty,
            "usable_96h": usable,
        })
    columns = ["sid", "season", "basin", "tropical_start", "et_time",
               "has_et_report", "usable_96h"]
    return pd.DataFrame(storms, columns=columns).set_index("sid")


if __name__ == "__main__":
    tracks = read_tracks()
    storms = storm_catalog(tracks)
    summary = pd.DataFrame(index=pd.Index(BASINS, name="basin"))
    by_basin = storms["basin"]
    summary["total_tcs"] = by_basin.value_counts().reindex(BASINS, fill_value=0)
    summary["et_tcs"] = storms["et_time"].notna().groupby(by_basin).sum().reindex(BASINS, fill_value=0)
    total = summary["total_tcs"].replace(0, np.nan)
    summary["et_percentage"] = (100 * summary["et_tcs"] / total).round(2)
    usable = storms["usable_96h"].groupby(by_basin).sum()
    summary["et_usable_96h"] = usable.reindex(BASINS, fill_value=0)


    print(f"\n{summary.to_string()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "ibtracs_stats.csv"
    summary.to_csv(output)
    print(f"\nSaved {output}")
