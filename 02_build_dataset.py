"""Generate storm-level train, validation, and test sample lists."""

from importlib import import_module

import numpy as np
import pandas as pd

# Reuse the statistics script's track cleaning, ET definition, and history checks.
ibtracs = import_module("01_ibtracs_stats")
OUTPUT_DIR = ibtracs.OUTPUT_DIR
# At each forecast time, look back by HISTORY for inputs and ahead by HORIZON for the label.
STEP, HISTORY, HORIZON = ibtracs.STEP, ibtracs.HISTORY, ibtracs.HORIZON
SPLIT_END_SEASONS = {"train": 2015, "val": 2020, "test": 2025}
CLASS_1_HOURS = 24
SEED = 42
MAX_TRAIN_NEGATIVES_PER_STORM = 4
COLUMNS = [
    "sid", "season", "basin", "label", "prediction_time", "et_time",
    "history_start_time", "history_end_time", "split",
]


def reliable_negative_windows(reports, predictions):
    """Check whether USA reports support a Class 0 label for each forecast time.

    `reports` is the time-sorted DataFrame for one storm. `predictions` is a
    pandas DatetimeIndex of candidate forecast times. Return one boolean per
    candidate. True means reports cover the next 48 h, with known non-ET
    classifications and no report gap longer than MAX_REPORT_GAP.
    """
    times = pd.DatetimeIndex(reports["ISO_TIME"])
    # Use the last report at or before the forecast and the first at or after its horizon.
    start = times.searchsorted(predictions, side="right") - 1
    end = times.searchsorted(predictions + HORIZON)
    covered = (start >= 0) & (end < len(times))
    start_indices, end_indices = start[covered], end[covered]

    # Compare cumulative counts at the window boundaries to detect any bad status or gap.
    uncertain = ~reports["USA_STATUS"].isin(ibtracs.NON_ET_STATUSES)
    bad_report_count = np.r_[0, uncertain.cumsum().to_numpy()]
    big_gap_count = times.to_series().diff().gt(ibtracs.MAX_REPORT_GAP).cumsum().to_numpy()
    valid = np.zeros(len(predictions), dtype=bool)
    valid[covered] = (
        (bad_report_count[end_indices + 1] == bad_report_count[start_indices])
        & (big_gap_count[end_indices] == big_gap_count[start_indices])
    )
    return valid


def sample_training_negatives(pool, target):
    """Select up to `target` Class 0 rows from the training candidates in `pool`.

    The pool contains a storm ID and a source (early ET or non-ET) for each
    row. Aim for an even mix of sources and limit how often each storm appears.
    The returned DataFrame contains the selected rows.
    """
    rng = np.random.default_rng(SEED)
    sources = ("early ET", "non-ET")
    quotas = (target // 2, target - target // 2)
    ranked = {}
    picked = {}
    for source, quota in zip(sources, quotas):
        rows = pool[pool["negative_source"].eq(source)]
        rows = rows.iloc[rng.permutation(len(rows))].copy()
        rows["storm_rank"] = rows.groupby("sid").cumcount()
        # Give each storm one chance in each round, up to the per-storm cap.
        rows = rows[rows["storm_rank"].lt(MAX_TRAIN_NEGATIVES_PER_STORM)]
        ranked[source] = rows.sort_values("storm_rank", kind="stable")
        picked[source] = min(quota, len(ranked[source]))
    # If one source falls short, take additional rows from the other where available.
    for source in sources:
        remaining = target - sum(picked.values())
        picked[source] += min(remaining, len(ranked[source]) - picked[source])
    if sum(picked.values()) < target:
        print(f"Training balance shortfall: found {sum(picked.values())} of {target} "
              f"class 0 samples with a cap of {MAX_TRAIN_NEGATIVES_PER_STORM} per SID.")
    if any(picked[source] < quota for source, quota in zip(sources, quotas)):
        print("One or both class 0 sources had fewer candidates than their target share.")
    print(f"Training class 0 sources: {picked}")
    return pd.concat([ranked[source].head(picked[source]) for source in sources])


if __name__ == "__main__":
    tracks = ibtracs.read_tracks()
    storms = ibtracs.storm_catalog(tracks)
    candidates = []
    for sid, reports in tracks.groupby("SID", sort=False):
        storm = storms.loc[sid]
        season, et_time = storm["season"], storm["et_time"]
        split = next(name for name, last in SPLIT_END_SEASONS.items() if season <= last)
        if pd.notna(et_time):
            # Use the last grid time strictly before the actual EX report.
            last = et_time.ceil(STEP) - STEP
        elif storm["has_et_report"]:
            # These storms have evidence of ET but an unresolved event time.
            continue
        else:
            last = (reports["ISO_TIME"].iloc[-1] - HORIZON).floor(STEP)
        # The first cutoff needs a full history starting at or after the first tropical report.
        first = (storm["tropical_start"] + HISTORY).ceil(STEP)
        if first > last:
            continue
        predictions = pd.date_range(first, last, freq=STEP)
        # Lead time is the time remaining until the reported ET (NaT for non-ET storms).
        lead = et_time - predictions
        positive = storm["usable_96h"] & (lead <= HORIZON)
        negative = reliable_negative_windows(reports, predictions)
        if pd.notna(et_time):
            negative &= lead > HORIZON
        valid = (positive | negative) & ibtracs.tropical_cutoffs(reports, predictions)
        predictions = predictions[valid]
        if predictions.empty:
            continue

        # A complete input window has both coordinates at all nine timestamps.
        histories = ibtracs.position_histories(reports, predictions)
        complete = (histories["latitude"].notnull() & histories["longitude"].notnull())
        predictions = predictions[complete.all(dim="history_step").values]
        lead = et_time - predictions
        # Start with Class 0, assign Class 2 within 48 h, then Class 1 within 24 h.
        label = np.zeros(len(predictions), dtype=int)
        label[lead <= HORIZON] = 2
        label[lead <= pd.Timedelta(hours=CLASS_1_HOURS)] = 1
        candidates.append(pd.DataFrame({
            "sid": sid, "season": season, "basin": storm["basin"], "label": label,
            "prediction_time": predictions, "et_time": et_time,
            "history_start_time": predictions - HISTORY, "history_end_time": predictions,
            "split": split, "negative_source": "non-ET" if pd.isna(et_time) else "early ET",
        }))

    samples = (pd.concat(candidates, ignore_index=True) if candidates
               else pd.DataFrame(columns=COLUMNS + ["negative_source"]))
    train = samples["split"].eq("train")
    positive = samples["label"].ne(0)
    # Match the training Class 0 count to Class 1 and keep every validation/test candidate.
    target = int((train & samples["label"].eq(1)).sum())
    negatives = sample_training_negatives(samples[train & ~positive], target)
    samples = pd.concat([samples[~train | positive], negatives], ignore_index=True)
    samples = samples.sort_values(["season", "sid", "prediction_time"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_END_SEASONS:
        path = OUTPUT_DIR / f"{split}_samples.csv"
        samples.loc[samples["split"].eq(split), COLUMNS].to_csv(
            path, index=False, date_format="%Y-%m-%d %H:%M:%S"
        )
        print(f"Saved {path}")

    counts = samples.groupby(["split", "basin", "label"]).size()
    rows = pd.MultiIndex.from_product(
        [SPLIT_END_SEASONS, ibtracs.BASINS], names=["split", "basin"]
    )
    table = counts.unstack(fill_value=0).reindex(index=rows, columns=[0, 1, 2], fill_value=0)
    print(f"\nSample counts by class, basin, and split:\n{table.to_string()}")
    print(f"\nTotals by split:\n{table.groupby(level='split', sort=False).sum().to_string()}")
