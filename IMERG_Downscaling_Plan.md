# IMERG 10 km → 1 km Precipitation Downscaling Plan

## Project Objective
Downscale IMERG precipitation data from ~10 km to 1 km resolution using machine learning regression with AORC as validation ground truth. Incorporate NLCD elevation/land cover as auxiliary predictors.

---

## Phase 1: Data Acquisition & Preprocessing

### 1.1 Data Sources & Access

**IMERG (Integrated Multi-satellitE Retrievals for GPM)**
- Source: NASA GES DISC
- Access: earthdata.nasa.gov credentials required
- Resolution: ~10 km global
- Format: HDF5 (.he5)
- Variable: `precipitationCal` (calibrated precipitation)
- Suggested temporal range: 2015-2020 (covers AORC availability)
- Download strategy: Use `earthdata` or `opendap` library; batch download monthly files

**AORC (NLDAS-2 forcing data regridded)**
- Source: NOAA/NWS OWP (Open Water Partnership)
- Access: https://data.nwis.usgs.gov/api/ or AWS S3
- Resolution: 1 km (actually 1/120 degree ≈ 925 m)
- Format: NetCDF
- Variable: `Rainf` (precipitation rate, convert to daily total)
- Temporal range: 2015-2020 (match IMERG)
- Download strategy: Direct NetCDF download or S3 sync

**NLCD (National Land Cover Database)**
- Source: USGS
- Access: USGS/MRLC via GEE or direct download
- Resolution: 30 m native → resample to 1 km
- Variables needed:
  - Elevation (DEM from NLCD or USGS 3DEP)
  - Land cover class (categorical: urban, forest, water, etc.)
  - Percent impervious surface
- Coverage: CONUS (adjust domain as needed)
- Download strategy: Google Earth Engine for ease, or USGS direct

### 1.2 Directory Structure

```
project_root/
├── data/
│   ├── raw/
│   │   ├── imerg/
│   │   │   ├── 2015/  (monthly .he5 files)
│   │   │   └── ...
│   │   ├── aorc/
│   │   │   ├── 2015.nc
│   │   │   └── ...
│   │   └── nlcd/
│   │       ├── dem_1km.tif
│   │       ├── lulc_1km.tif
│   │       └── impervious_1km.tif
│   ├── processed/
│   │   ├── imerg_coarse.nc        (10 km, daily mean)
│   │   ├── aorc_fine.nc           (1 km, daily total)
│   │   ├── features_train.nc      (training features)
│   │   └── features_test.nc       (test features)
│   └── auxilliary/
│       └── dem_slope_aspect.nc    (derived from DEM)
├── models/
│   ├── downscaler_rf.pkl          (trained Random Forest)
│   └── model_metadata.json        (hyperparams, feature importance)
├── results/
│   ├── validation_metrics.json
│   ├── spatial_plots/
│   └── timeseries_comparisons/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_results_analysis.ipynb
└── src/
    ├── data_pipeline.py           (download, parse, grid-align)
    ├── feature_engineering.py     (auxiliary + coarse IMERG features)
    ├── train_model.py             (RF training)
    ├── predict.py                 (apply to full IMERG dataset)
    ├── validation.py              (evaluate vs AORC)
    └── utils.py                   (grid alignment, metrics, viz)
```

### 1.3 Data Preprocessing Pipeline

**Step 1: Download & Parse IMERG**
- Extract `precipitationCal` from .he5 files
- Resample to daily totals (sum half-hourly)
- Reproject to consistent lat/lon grid (WGS84)
- Subset to CONUS or region of interest
- Output: NetCDF `imerg_raw_10km.nc` (var: precip, dims: time × lat × lon)

**Step 2: Download & Parse AORC**
- Extract `Rainf` (kg/m²/s)
- Convert to daily total (sum over 24 hrs, units: mm/day)
- Reproject to standard 1 km grid
- Output: NetCDF `aorc_1km_daily.nc`

**Step 3: Download & Process NLCD**
- DEM: Get 30 m native, resample to 1 km (bilinear)
- LULC: 30 m native, resample to 1 km (mode/nearest)
- Derived: Calculate slope, aspect from DEM (using `richdem` or `rasterio`)
- Impervious: If available, resample to 1 km
- Output: NetCDF `nlcd_features_1km.nc` (vars: dem, lulc, slope, aspect, imperv)

**Step 4: Grid Alignment**
- Coarsen AORC from 1 km → 10 km (mean aggregation)
  - This becomes training target at coarse scale
- Reproject IMERG to exact same 10 km grid as coarsened AORC
- Reproject NLCD features to 1 km grid, matching AORC exactly
- Ensure all grids are perfectly aligned (same lat/lon bounds, same cell centers)
- Output: 
  - `imerg_aligned_10km.nc`
  - `aorc_aligned_10km.nc` (coarsened)
  - `aorc_aligned_1km.nc` (kept at 1 km)
  - `nlcd_aligned_1km.nc`

---

## Phase 2: Feature Engineering

### 2.1 Coarse-Scale Features (10 km)

From IMERG at 10 km:
- Precipitation itself (raw)
- 7-day rolling mean (temporal context)
- Precipitation percentile rank (in local 10-year window)
- Spatial mean in 3×3 neighborhood

From NLCD coarsened to 10 km:
- Mean elevation
- Dominant land cover class (one-hot encoded)
- Mean slope
- Urban fraction (% impervious)

### 2.2 Fine-Scale Features (1 km)

From NLCD at 1 km:
- Elevation (DEM)
- Slope
- Aspect (sin/cos transformed)
- Land cover class (one-hot or ordinal)
- Impervious fraction

Interaction terms:
- Elevation × slope (orographic factor proxy)
- Elevation × land_cover

### 2.3 Training & Validation Data Assembly

**Temporal split (critical—no spatial leakage):**
- Train: 2015-01-01 to 2018-12-31 (1461 days)
- Validation: 2019-01-01 to 2020-12-31 (730 days)

**Training dataset construction:**
```
For each day in training period:
  - Flatten 10 km IMERG grid → vector of ~1000 cells
  - Flatten 10 km NLCD (coarsened) features → vectors
  - Stack into feature matrix X_train (size: 1461*1000 × n_features)
  
  - Flatten coarsened AORC (target) → vector
  - Stack into target vector y_train (size: 1461*1000,)
```

**Validation dataset:**
- Same structure, 2019-2020 dates
- Will evaluate downscaled 1 km predictions against 1 km AORC

Output:
- `X_train.npy` or `.h5` (large; ~2-3 GB)
- `y_train.npy` (~10 MB)
- `X_val.npy`
- `y_val.npy`
- Store feature names & grid metadata separately (`.json`)

---

## Phase 3: Model Development

### 3.1 Model Architecture: Ensemble Tree-Based Regression

**Primary model: Gradient Boosted Trees (XGBoost or LightGBM)**

Rationale:
- Handles non-linear relationships (elevation → precip)
- Captures feature interactions (slope × elevation)
- Computationally efficient for large spatial datasets
- Provides feature importance ranking

**Hyperparameter baseline:**
```python
n_estimators: 200
max_depth: 6
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
min_child_weight: 5
```

**Fallback: Random Forest** (simpler, more stable)
```python
n_estimators: 200
max_depth: 12
min_samples_split: 20
```

### 3.2 Training Procedure

1. **Hyperparameter tuning** (optional, time-consuming):
   - Use validation set for early stopping (if GBM)
   - OR 5-fold cross-validation on train subset (sample ~50% for speed)
   
2. **Train on full training set** with tuned hyperparams

3. **Feature importance ranking**:
   - Extract permutation importance or SHAP values
   - Document top 10 predictive features

4. **Save checkpoint**:
   - Pickle model + metadata (feature names, scaling params if any)

### 3.3 Expected Training Details

- Training time: ~1-5 minutes on CPU (depending on dataset size)
- Memory: ~4-8 GB RAM
- Model size: ~500 MB (pickle)

---

## Phase 4: Downscaling & Prediction

### 4.1 Apply Model to Full IMERG Timeline

**For each day in 2015-2020:**
1. Fetch IMERG 10 km grid for that day
2. Compute coarse + fine-scale features
3. Reshape to model input (10 km cells as rows)
4. Run model prediction → output at 10 km resolution
5. **Upsample predictions to 1 km**:
   - For each 10 km cell, use spatial interpolation (bilinear) + NLCD gradient correction:
     - Estimate sub-grid variability using fine-scale NLCD features
     - Multiply predicted 10 km value by local factor (elevation gradient, slope-adjusted)
   - OR: Train secondary local model per 10 km cell (local residual corrections)

### 4.2 Upsampling Strategy (Detailed)

**Simple approach (fast):**
- Bilinear interpolation from 10 km → 1 km
- Multiply by elevation lapse-rate correction:
  ```
  downscaled_precip = pred_10km * (1 + α * (DEM_1km - DEM_10km_mean))
  α ≈ -0.0001 to -0.0003 (typical precip lapse rate: -0.5 mm/100 m)
  ```

**Advanced approach (slower):**
- Train local 1 km correction model:
  - Input: residuals (observed AORC − predicted 10 km upsampled)
  - Train local RF on 1 km NLCD features
  - Apply corrections to full timeline
- Blends model predictions with topographic physics

### 4.3 Output

- NetCDF: `imerg_downscaled_1km_2015_2020.nc`
  - Dims: time × lat × lon (1 km resolution)
  - Vars: precipitation (mm/day), uncertainty (std of local residuals)
  - Metadata: model version, training dates, feature list

---

## Phase 5: Validation Framework

### 5.1 Validation Metrics (at 1 km on 2019-2020)

**Aggregate statistics:**
- RMSE (overall)
- Bias (mean error)
- Pearson correlation
- Nash-Sutcliffe Efficiency (NSE)
- KGE (Kling-Gupta Efficiency)

**Stratified by intensity:**
- Dry days (precip < 1 mm): bias, correlation
- Light (1-10 mm): RMSE, NSE
- Moderate (10-30 mm): RMSE, NSE
- Heavy (>30 mm): RMSE, POD (Probability of Detection)

**Spatial skill:**
- Spatial correlation map (per 3-month window)
- Quantile-quantile plots (empirical distributions)

**Temporal skill:**
- Time series comparison at 5-10 sample locations (e.g., major cities)
- Seasonal breakdown (JFMAMJJASOND)

### 5.2 Reference Comparisons

Run baseline method for comparison:
- **Bilinear interpolation only**: 10 km IMERG → 1 km (no ML)
  - Expected: lower skill than ML, especially in complex terrain
  
Compare:
| Metric | ML Downscaled | Bilinear Baseline | IMERG orig (upsampled) |
|--------|---------------|--------------------|------------------------|
| RMSE   |               |                    |                        |
| NSE    |               |                    |                        |
| Bias   |               |                    |                        |

### 5.3 Validation Code Structure

```python
# metrics_dict = validate(downscaled_1km, aorc_1km_val, nlcd_features)
# Outputs: JSON with all metrics above
# Generates: spatial maps, timeseries plots, quantile plots
```

---

## Phase 6: Implementation Checklist

### Code Modules to Build (in `src/`)

- [ ] **data_pipeline.py**
  - `download_imerg(start_date, end_date)` → raw .he5 files
  - `parse_imerg(file_path)` → xr.Dataset at 10 km
  - `download_aorc(start_date, end_date)` → NetCDF
  - `download_nlcd()` → DEM, LULC, impervious
  - `align_grids(imerg, aorc, nlcd)` → all projected, matched
  - `aggregate_daily(ds, var)` → daily means/totals

- [ ] **feature_engineering.py**
  - `coarse_features_10km(imerg)` → DataFrame at 10 km
  - `fine_features_1km(nlcd)` → DataFrame at 1 km
  - `prepare_training_data(imerg_10km, aorc_10km, nlcd_1km, date_split)` → X_train, y_train, X_val, y_val
  - `compute_derived_features(dem, lulc)` → slope, aspect, etc.

- [ ] **train_model.py**
  - `train_model(X_train, y_train, model_type='xgb', hyperparams=None)` → fitted model object
  - `get_feature_importance(model)` → DataFrame
  - `save_model(model, path)` → pickle

- [ ] **predict.py**
  - `downscale_day(imerg_day_10km, nlcd_1km, trained_model)` → predictions at 10 km
  - `upsample_to_1km(pred_10km, nlcd_features, method='bilinear+lapse')` → 1 km output

- [ ] **validation.py**
  - `compute_metrics(downscaled_1km, aorc_1km, by_intensity=False)` → dict of metrics
  - `plot_spatial_comparison(pred, obs, title)` → map figure
  - `plot_timeseries_sample(pred, obs, locations)` → time series plot
  - `quantile_quantile_plot(pred, obs)` → Q-Q plot

- [ ] **utils.py**
  - `align_grids()`, `reproject_raster()`, `flatten_to_array()`, etc.
  - `load_config()`, logging setup

### Main Execution Script

**`main.py` or Jupyter notebook:**
```python
# 1. Download & preprocess all data
data = setup_data(year_start=2015, year_end=2020)

# 2. Prepare training data
X_train, y_train, X_val, y_val = prepare_training_data(data, date_split='2019-01-01')

# 3. Train model
model = train_model(X_train, y_train, model_type='xgb')

# 4. Downscale full time series
for year in 2015:2020:
    downscaled_1km[year] = downscale_year(data[year], model, nlcd_features)

# 5. Validate
metrics = validate(downscaled_1km, aorc_1km_2019_2020)
print_results(metrics)
```

---

## Phase 7: Expected Outputs & Success Criteria

### Outputs

1. **Downscaled Dataset**: NetCDF file, 1 km resolution, 2015-2020 daily precipitation
2. **Validation Report**: JSON with all metrics, spatial/temporal breakdowns
3. **Figures**:
   - Spatial maps: bias, RMSE (2019-2020 average)
   - Time series at 5 locations
   - Quantile-quantile plots
   - Feature importance bar chart

### Success Criteria

- **RMSE at 1 km**: < 20% reduction vs. simple bilinear interpolation
- **Spatial skill**: NSE > 0.6 at grid cells (especially complex terrain)
- **Extreme events**: Capture >80% of days with >30 mm precip
- **No spatial artifacts**: Downscaled field should not show artificial grid patterns

---

## Phase 8: Timeline & Compute Requirements

| Phase | Task | Duration | Notes |
|-------|------|----------|-------|
| 1 | Data download/preprocessing | 2-4 hrs | Mostly I/O bound; can parallelize downloads |
| 2 | Feature engineering | 1-2 hrs | Memory-intensive; may need chunking |
| 3 | Model training | 30-60 min | CPU or GPU accelerated if available |
| 4 | Downscaling full timeline | 1-3 hrs | Inference on 6 years × 365 days |
| 5 | Validation | 1 hr | Mostly I/O + plotting |
| **Total** | | **6-14 hrs** | Parallelizable; actual wall time ~3-6 hrs |

**Compute specs (minimum):**
- CPU: 4+ cores
- RAM: 16 GB (32 GB recommended)
- Storage: 50-100 GB (temporary + outputs)
- GPU: Optional (speeds up training 5-10×)

---

## Notes & Caveats

1. **Spatial leakage risk**: Train/val split *must* be temporal, not spatial
2. **Data quality**: IMERG has known issues in complex terrain & mountainous regions; downscaling won't fix missing data
3. **AORC availability**: Confirm 2015-2020 coverage for your region; may have gaps
4. **Upsampling physics**: Simple bilinear + lapse-rate is empirical; hybrid approach more robust
5. **Extreme events**: ML models may underpredict extremes; consider quantile regression if needed
6. **Validation domain**: Test on different regions if possible (e.g., train CONUS, test mountain subset)

---

## References & Tools

**Libraries:**
- `xarray`, `rasterio`, `rioxarray` — data I/O & regridding
- `scikit-learn`, `xgboost`, `lightgbm` — modeling
- `numpy`, `pandas` — array operations
- `richdem` or `scipy` — DEM derivatives
- `matplotlib`, `cartopy` — visualization

**Data APIs:**
- `earthdata` Python package — NASA data download
- `google.colab` + Earth Engine — NLCD via GEE
- Direct USGS/NOAA links

---

## Next Steps for Claude Code

1. Validate this plan for feasibility
2. Build Phase 1 data pipeline (download stub + parser)
3. Implement grid alignment & testing
4. Scale to full training data once alignment confirmed
5. Train & evaluate model
