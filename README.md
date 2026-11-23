# Tropical cyclone ET dataset

This project uses the local IBTrACS v04r01 CSV to count tropical cyclones and build
sample lists for extratropical transition (ET) prediction in WP, EP, and NA for
the 1980–2025 seasons.

## Dataset construction

Each sample represents a forecast at one cutoff on the 6-hour UTC grid. The
storm must be tropical at the cutoff, and its 48-hour input window must have
all nine USA positions, including the cutoff.

1. **Identify useful ET tracks.** A main track must reach at least tropical
   depression status in the USA reports, have a reliably identified ET time,
   and have at least 96 hours from its first tropical report to ET. It must
   also have positions at every required 6-hour grid time in that pre-ET
   history and a tropical report at each of the eight positive cutoffs.
2. **Generate positive samples.** Each useful ET track contributes four
   Class 1 cutoffs with ET within 24 hours and four Class 2 cutoffs with ET
   more than 24 and up to 48 hours ahead. 
3. **Build the Class 0 pool.** Use eligible cutoffs more than 48 hours before
   a confirmed ET or on tracks with no recorded ET. Each cutoff needs USA
   reports that verify an ET-free 48-hour horizon. 
4. **Split by season and balance training.** Keep all samples from a SID
   together. Keep every valid cutoff in validation and test. In training,
   sample Class 0 with seed 42 to match the count of each positive class,
   aiming for half from early cutoffs on ET tracks and half from non-ET
   tracks, with at most four training negatives per SID. 

## Sample distribution

Counts from the current sample CSVs in `data/processed/`.

| Split | Basin | Class 0 | Class 1 | Class 2 | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | WP | 329 | 396 | 396 | 1,121 |
| Train | EP | 299 | 56 | 56 | 411 |
| Train | NA | 344 | 520 | 520 | 1,384 |
| **Train subtotal** | All | **972** | **972** | **972** | **2,916** |
| Validation | WP | 1,130 | 136 | 136 | 1,402 |
| Validation | EP | 763 | 8 | 8 | 779 |
| Validation | NA | 804 | 112 | 112 | 1,028 |
| **Validation subtotal** | All | **2,697** | **256** | **256** | **3,209** |
| Test | WP | 872 | 84 | 84 | 1,040 |
| Test | EP | 469 | 0 | 0 | 469 |
| Test | NA | 605 | 88 | 88 | 781 |
| **Test subtotal** | All | **1,946** | **172** | **172** | **2,290** |
| **Overall** | All | **5,615** | **1,400** | **1,400** | **8,415** |

## Run

Use a Python environment with pandas, NumPy, and xarray. Run both scripts from
this directory.

```bash
python 01_ibtracs_stats.py
python 02_build_dataset.py
```

The scripts read `data/raw/Ibtracks/ibtracs.ALL.list.v04r01.csv` and use paths
relative to the project directory. Edit the constants at the top of the scripts
to change settings. The first script writes `ibtracs_stats.csv`. The second writes
`train_samples.csv`, `val_samples.csv`, and `test_samples.csv`. All four are in
`data/processed/`.

## Storm and ET definitions

- Use rows with `TRACK_TYPE == "main"` and a populated `USA_AGENCY`.
  Drop rows with invalid timestamps.
- Count a SID once if it reaches tropical depression or stronger in
  `USA_STATUS` (`TD`, `TS`, `HU`, `HR`, `TY`, `ST`, or `TC`). Assign its
  basin at the first qualifying report and retain its full USA track.
- A reliable ET time is the first `EX` in `USA_STATUS` after tropical genesis
  when the preceding USA report is tropical and at most six hours earlier.
  Exclude storms with an unresolved `EX` time from sampling.
- Pre-ET lifetime is elapsed time from the first tropical USA report to ET.
  A usable 96-hour history has observations at every 6-hour grid time from
  the grid time at or before ET−96 h through the last grid time before ET,
  with a tropical USA classification at all eight positive cutoffs.

## Position histories

Forecasts use a 6-hour grid at 00, 06, 12, and 18 UTC. Each 48-hour input
window has nine `USA_LAT` and `USA_LON` positions, including the prediction
time. In the local CSV, the pre-interpolated USA positions have an empty
`USA_AGENCY`, including a few positions on the 6-hour grid. Those positions
are treated as gaps. Any window with a missing position is discarded. The
actual report time is used for ET, including `EX` reports between grid times.

## Labels and splits

| Label | Meaning |
| --- | --- |
| 0 | The following 48 hours are verified ET-free. |
| 1 | ET occurs within 24 hours, including the 24-hour boundary. |
| 2 | ET occurs more than 24 hours and up to 48 hours ahead. |

Each cutoff must have a tropical USA report at that time. Class 0 requires
tropical, subtropical, or disturbance classifications through the first report
at or after the horizon, with no gap greater than six hours. Tracks whose
follow-up ends too early are excluded for that cutoff.

Train covers 1980–2015, validation 2016–2020, and test 2021–2025. All samples from
a SID stay together. Each eligible ET storm contributes the four 6-hour-grid
cutoffs in each positive class. For an on-grid ET, the lead times are 6, 12,
18, and 24 hours for class 1 and 30, 36, 42, and 48 hours for class 2. Training
class 0 targets an even mix of early periods of ET storms and non-ET storms
using seed 42 and at most four negative samples per SID. Validation and test
keep every valid 6-hour cutoff without class balancing.

The sample CSVs contain UTC timestamps and window metadata. Keep `et_time` and
`label` for supervision and evaluation. Model inputs must use data available
through `history_end_time`.

[NOAA IBTrACS v04r01 column documentation](https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf)
