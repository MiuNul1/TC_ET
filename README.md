# Tropical cyclone ET dataset

Predict extratropical transition (ET) in the western North Pacific (WP), eastern
North Pacific (EP), and North Atlantic (NA) using IBTrACS v04r01 and ERA5.

## Dataset construction

- **Identify storms with recorded ET and usable histories.**
  - The storm reaches tropical depression strength or higher in IBTrACS
    main-track USA reports.
  - At least 96 hours elapse from the first tropical report to ET. Positions
    are available at every required 6-hour time in the pre-ET history, and the
    storm is at least a tropical depression at all eight forecast times in the
    final 48 hours before ET.
- **Assign labels.**
  - **Class 0.** USA reports verify an ET-free 48-hour forecast horizon.
  - **Class 1.** ET occurs within 24 hours after the forecast time.
  - **Class 2.** ET occurs more than 24 hours and up to 48 hours after the forecast time.
  - Each eligible ET storm contributes four Class 1 and four Class 2 samples.
- **Split and balance.**
  - Split by season and keep each storm in one split. Train on 1980–2015,
    validate on 2016–2020, and test on 2021–2025.
  - Each eligible ET storm contributes four Class 1 and four Class 2 samples.
  - Keep all eligible validation and test samples and all Class 1 and Class 2
    training samples from storms with confirmed ET.
    Sample training Class 0 candidates to match the Class 1 count.
  - Aim to draw half of the training Class 0 samples from forecast times more
    than 48 hours before confirmed ET and half from storms with no recorded ET.

## Scripts

### `01_ibtracs_stats.py`

- Reads `data/raw/Ibtracks/ibtracs.ALL.list.v04r01.csv` and builds a storm catalog.
- Summarizes cyclone counts, ET counts and percentages, and usable ET tracks by basin.
- Writes `data/processed/ibtracs_stats.csv`.

### `02_build_dataset.py`

- Reuses the storm catalog and track filters from `01_ibtracs_stats.py`.
- Records each sample's storm ID, label, forecast time, and history bounds.
- Writes `train_samples.csv`, `val_samples.csv`, and `test_samples.csv` under `data/processed/`.

### `03_extract_features.py`

- Extracts latitude, heading sine and cosine, and central pressure from IBTrACS.
- Computes 200–850 hPa wind shear and mean ocean SST within 500 km using ERA5.
- Writes `data/processed/features/{train,val,test}_features.csv` with one row per sample and history time.

## Sample distribution

| Split | Class 0 | Class 1 | Class 2 | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 972 | 972 | 972 | 2,916 |
| Validation | 2,697 | 256 | 256 | 3,209 |
| Test | 1,946 | 172 | 172 | 2,290 |
| **Total** | **5,615** | **1,400** | **1,400** | **8,415** |
