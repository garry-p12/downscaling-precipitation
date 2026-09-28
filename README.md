# IMERG 10 km → 1 km precipitation downscaling

Machine-learning downscaling of NASA IMERG daily precipitation (0.1°, ~10 km)
to the 1 km AORC grid, using NLCD land cover, impervious fraction and a DEM as
auxiliary predictors and AORC as the training target / validation truth.
Implements [IMERG_Downscaling_Plan.md](IMERG_Downscaling_Plan.md).

Default domain: **Austin, TX region** (`[-99.0, 28.5, -96.0, 32.0]` — Hill
Country / Balcones Escarpment through the Blackland Prairie; San Antonio to
Waco), 2015-2020, train 2015-2018, validate 2019-2020.

## How it works

```
IMERG 10 km ──┐                      ┌─ stage 1 (XGBoost, 10 km) ─┐
NLCD/DEM  → 10 km aggregates ────────┤   target: AORC block mean  │
AORC 1 km → 10 km block mean (target)┘                            │
                                                                  ▼
                       bilinear upsample + 1 km NLCD/DEM features ─┤
                       stage 2 (XGBoost, 1 km residual model)     │
                       target: AORC 1 km − upsampled stage-1      ▼
                                                     1 km precipitation (mm/day)
```

* **Stage 1** corrects IMERG at its native resolution (bias, intensity,
  neighbourhood context, seasonality, terrain/land-cover aggregates).
* **Stage 2** (`upsampling.method: residual`) learns sub-grid structure from
  1 km elevation anomaly, slope/aspect, land-cover fractions and impervious
  fraction. Alternatives: `bilinear`, `bilinear+lapse` (elevation lapse-rate
  factor), optional `conserve_mass` rescaling.
* **Validation** compares the ML product with two baselines on identical
  samples: bilinear IMERG and nearest-neighbour IMERG (the original data).

Grid convention: the fine grid *is* the native AORC 1/120° grid (AORC is never
interpolated); each IMERG 0.1° cell maps to a 12×12 block. See `src/utils.py`.

## Setup

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate downscale
./run_tests.sh                           # 29 tests (pipeline + deep), see pytest.ini
```

The two test suites run as separate processes because torch's bundled OpenMP
runtime and the one behind rasterio/xgboost cannot coexist in one process on
macOS; `pytest -q tests` alone runs the pipeline suite.

### Earthdata credentials (IMERG only)

AORC, NLCD and the DEM are anonymous. IMERG needs a free NASA Earthdata login:

1. Create an account: https://urs.earthdata.nasa.gov/users/new
2. Approve the **"NASA GESDISC DATA ARCHIVE"** application at
   https://urs.earthdata.nasa.gov/approved_applications (required, otherwise
   downloads return 401).
3. Provide the credentials to `earthaccess` either via `~/.netrc`

   ```
   machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASS
   ```
   (`chmod 600 ~/.netrc`), or via environment variables
   `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`.

## Running

```bash
python main.py download --nlcd     # Copernicus GLO-90 DEM + MRLC NLCD (WCS)   ~3 min
python main.py download --aorc     # AORC hourly Zarr on AWS -> daily 1 km    ~15-30 min
python main.py download --imerg    # GES DISC OPeNDAP subsets (Earthdata)     ~10-20 min
python main.py preprocess          # parse granules, align all grids
python main.py features            # stage-1 matrices (X_train.npy ...)
python main.py train               # stage 1 + stage 2
python main.py predict             # results/imerg_downscaled_1km_2015_2020.nc
python main.py validate            # results/validation_metrics.json + figures
```

`python main.py all` runs everything; `python main.py all --synthetic`
generates synthetic raw data in the real file formats and runs the complete
pipeline without any downloads (useful for testing on a laptop).

All steps are resumable (downloads and yearly outputs skip existing files) and
all settings live in [config.yaml](config.yaml). Raw IMERG files are deleted
after parsing unless `data.imerg.keep_raw: true`.

IMERG is fetched by default through GES DISC's OPeNDAP server, which subsets
the granule to the domain server-side (a few KB per day instead of 28 MB).
Set `data.imerg.mode: granule` to download the full daily granules with
`earthaccess` instead (same credentials).

If you already have IMERG granules (e.g. from the GES DISC subsetter or
`wget`), drop them under `data/raw/imerg/` and run `main.py preprocess`; both
daily (`3B-DAY`) and half-hourly (`3B-HHR`) files are recognised.

## Layout

```
config.yaml                    domain, time split, data sources, model hyper-parameters
main.py                        CLI driver
src/utils.py                   config, grids, coarsen / upsample, NetCDF I/O
src/data_pipeline.py           IMERG / AORC / NLCD / DEM download, parse, align
src/feature_engineering.py     terrain derivatives, 10 km & 1 km features, training sets
src/train_model.py             XGBoost / RF training, importance, model I/O
src/predict.py                 stage-1 prediction, upsampling methods, full-timeline output
src/validation.py              streaming metrics, baselines, success criteria, figures
src/synthetic.py               synthetic raw data generator (tests / dry runs)
src/deep/                      CNN / Swin transformer / residual-diffusion downscalers
src/model_comparison.py        scores every product together (spectra, FSS, CRPS, ...)
slurm/                         TACC Vista job scripts (GH200)
scripts/vista_sync.sh          push code/data to the cluster, pull results back
tests/                         pytest suite
notebooks/                     01_data_exploration, 02_results_analysis
data/{raw,processed,auxiliary} inputs and aligned products (git-ignored)
models/                        downscaler_coarse.pkl, downscaler_fine.pkl, model_metadata*.json
results/                       downscaled NetCDF, validation_metrics.json, validation_report.md, figures
```

## Outputs

* `results/imerg_downscaled_1km_2015_2020.nc` — `precipitation(time, lat, lon)`
  in mm/day plus `uncertainty(lat, lon)` (std of 1 km residuals over the
  training period); metadata lists model versions, periods and feature lists.
* `results/validation_metrics.json` — overall / by-intensity / seasonal /
  quarterly / per-cell statistics for ML, bilinear and nearest, the comparison
  table from plan §5.2 and the success criteria from plan §7.
* Figures: bias/RMSE/NSE maps, wettest-day comparison, quarterly spatial
  correlation, Q-Q plot, seasonal metrics, time series at the configured
  cities, feature importance.

## Results — Austin domain, real data (validation 2019-2020, 1 km vs AORC)

Full run: IMERG via OPeNDAP (22 min), AORC (5 min), NLCD/DEM (3 min), train
(3 min), predict (2 min), validate (1 min). `results/validation_report.md`
and `results/validation_metrics.json` hold the complete numbers.

| Product | RMSE | MAE | bias | NSE | KGE | CRPS | POD>30mm | spectral ratio |
|---|---|---|---|---|---|---|---|---|
| IMERG nearest | 4.994 | 1.529 | −0.080 | 0.647 | 0.787 | 1.529 | 0.600 | 32.7 |
| Bilinear | 4.919 | 1.506 | −0.082 | 0.657 | 0.787 | 1.506 | 0.601 | 0.125 |
| **XGBoost 2-stage** | **4.583** | **1.467** | −0.046 | **0.702** | 0.739 | 1.467 | 0.548 | 0.021 |
| CNN (U-Net) | 4.663 | 1.511 | 0.044 | 0.692 | 0.784 | 1.511 | 0.614 | 0.162 |
| **Swin transformer** | 4.653 | 1.520 | 0.145 | 0.693 | **0.790** | 1.520 | **0.633** | 0.109 |
| Diffusion (ens. mean) | 4.807 | 1.493 | −0.113 | 0.673 | 0.746 | **1.050** | 0.511 | 0.248 |
| Diffusion (1 member) | 5.483 | 1.668 | −0.132 | 0.574 | 0.731 | 1.668 | 0.467 | **0.840** |

What this says:

* **No product wins everything, and that is the result.** XGBoost takes accuracy
  (RMSE 4.583, 6.8 % below bilinear); Swin takes extremes and variability (POD
  0.633, CSI 0.469, KGE 0.790 — the only product beating bilinear's KGE); the
  diffusion ensemble takes the probabilistic score (CRPS 1.050, 28 % better than
  any deterministic field) and its single member is the only one with realistic
  texture (84 % of observed sub-10 km variance).
* **Gains came from information, never architecture.** Four model families span
  4.583–4.807. What moved the numbers was ERA5 predictors (−2.2 %), a 4.6×
  longer training record (−2.0 %) and sub-daily structure from the IMERG daily
  counters (−1.5 %).
* **The 1 km residual stage is rejected every time** (hold-out 5.167 vs bilinear
  5.176), with 18.5 years of data and 31 features. Sub-grid placement is not
  recoverable from these inputs in this domain.
* **Plan targets missed**: 6.8 % RMSE reduction against a 20 % target, POD 0.633
  against 0.80. The domain is flat — 8 of 151,200 cells exceed 5° slope — and
  IMERG is already nearly unbiased here. Those targets assume orographic
  structure this domain does not have.

See [docs/PROJECT_EXPLAINER.md](docs/PROJECT_EXPLAINER.md) for the full account
and [docs/METRICS.md](docs/METRICS.md) for why RMSE alone is misleading here.

## Notes on deviations from the plan

* NLCD land cover enters as **class fractions** (from 30 m data) at both 1 km
  and 10 km rather than only the dominant class; DEM is downsampled with
  `average` (configurable) because bilinear aliases at 10× reduction.
* Early stopping uses the last 15 % of the *training* days so that 2019-2020
  stays a clean test set (`model.coarse.early_stopping_split: val` restores
  the plan's behaviour).
* `dem_anom` is computed against a bilinearly upsampled 10 km DEM (continuous
  across block edges); a block-mean anomaly imprints the 10 km grid on the
  output. A `blockiness_ratio` metric checks this (≈1 = no artifacts).
* Day-of-year sin/cos and 10 km DEM standard deviation are added as cheap,
  standard predictors.
