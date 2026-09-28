# IMERG 10 km → 1 km precipitation downscaling — project explainer

Written for someone joining the project: what it does, where the data comes
from, what was tried, what worked, and what did not. For the metric definitions
in depth see [METRICS.md](METRICS.md); for figures see
[`results/figures/`](../results/figures/).

---

## 1. The problem

Satellite precipitation (NASA IMERG) is available globally at 0.1° (~10 km).
Many applications — flood modelling, small catchments, urban hydrology — need
~1 km. The question is whether machine learning can add genuine information at
that scale, or merely interpolate.

**Domain**: Austin, Texas — `[-99.0, 28.5, -96.0, 32.0]`, roughly 300 × 390 km.
**Period**: 2000-06-01 to 2020-12-31 (7,519 days; IMERG V07 daily begins June 2000).
**Reference**: AORC, a 1 km gridded analysis, treated as truth.

One 10 km cell contains a **12 × 12 block of 144 one-kilometre cells**. The
domain is 1,050 coarse cells → 151,200 fine cells. So the model must produce
143 numbers per cell that are not determined by the input, constrained only by
their average. Whether that is possible is the empirical question the whole
project answers.

---

## 2. Data collection

| dataset | role | source | resolution | how acquired |
|---|---|---|---|---|
| **IMERG** `GPM_3IMERGDF` V07 | predictor | NASA GES DISC | 0.1°, daily | OPeNDAP server-side subsetting |
| **AORC** v1.1 | target / truth | NOAA on AWS S3 | 1/120°, hourly | anonymous Zarr, summed to daily |
| **ERA5** | predictor | ARCO-ERA5 on GCS | 0.25°, hourly/6-hourly | anonymous Zarr |
| **NLCD 2021** | predictor | USGS MRLC | 30 m | WCS tiles |
| **Copernicus GLO-90 DEM** | predictor | AWS | 90 m | anonymous COG |

Only IMERG needs credentials (free NASA Earthdata login).

### Acquisition notes that mattered

* **IMERG via OPeNDAP** subsets the granule server-side: ~33 KB/day instead of
  a 28 MB global file. All 247 months download in ~45 minutes.
* **AORC** is hourly; daily sums use `min_count=20`, so a day missing more than
  four hours becomes NaN rather than a silent under-count.
* **MERRA-2's OPeNDAP endpoint is retired** (HTTP 410). ERA5 replaced it.
* **ARCO-ERA5 chunking matters.** The hourly store uses one *global* field per
  chunk, so subsetting a small window still transfers ~3 GB per variable-month
  (~15 h for seven variables). The 6-hourly store is ~7× cheaper but lacks
  CAPE, so the two are mixed: CAPE from hourly, the rest from 6-hourly.
* **The daily IMERG granule carries sub-daily counters** —
  `precipitation_cnt_cond` is the number of raining half-hours. That makes the
  half-hourly product (360,912 granules, ~33 h) unnecessary for structure
  features.
* **`randomError` is excluded.** V07 daily labels it mm/day but its median is
  ~5,000 mm/day, which is not physical, so it is not usable as an uncertainty.

### Grids

Everything is forced onto one pair of grids and asserted before use:

* **Coarse** = IMERG's native 0.1° grid, centres at `edge + 0.05` → 35 × 30.
* **Fine** = AORC's native 1/120° grid → 420 × 360.

The fine grid *is* AORC's own grid, so the 1 km truth is **never interpolated**.
The cost is that the 12 × 12 block sits 1/240° (~460 m) off the IMERG cell
centre — under 5 % of a coarse cell, and preferable to resampling the target.

---

## 3. Features

### Coarse (10 km) — 27, rising to 31 with sub-daily structure

**From IMERG (dynamic)**
`imerg`, `imerg_roll7` (7-day mean), `imerg_pct` (percentile against that cell's
own training-period climatology), `imerg_nbr3` (3×3 neighbourhood mean).

**Sub-daily structure** (from the daily granule's half-hour counters)
`imerg_wet_frac`, **`imerg_cond_intensity`** (total ÷ wet-hours × 0.5 — the
convective/stratiform discriminator), `imerg_mw_frac` (microwave vs IR-morphed
share, a retrieval-trust proxy), `imerg_prob_liquid`.

**From ERA5** — the atmospheric state IMERG cannot see
`cape_max`, `cape`, `tcwv`, `tcwv_max`, `t2m`, `wind_speed`, `wind_sin`,
`wind_cos`, `moist_flux`.

**Static, aggregated to 10 km**
`dem_mean`, `dem_std`, `slope_mean`, `imperv_mean`, eight `lc_frac_*`.

**Temporal** `doy_sin`, `doy_cos`.

### Fine (1 km) — 23

The upsampled coarse fields (`pred_bl`, `imerg_bl`, …) plus 1 km statics
(`dem`, `slope`, `aspect_sin/cos`, `imperv`, land-cover fractions) and
interactions (`dem_x_slope`, `dem_anom`, `dem_x_forest`, `dem_x_developed`).

Two design notes:

* **`imerg_pct`** turns "12 mm" into "a 97th-percentile day *here*", letting one
  model span a wet east and a dry west. It ranks third by importance.
* **`dem_anom`** is elevation minus the *bilinearly* upsampled block-mean
  elevation. A block-constant mean stamps the 10 km grid onto the 1 km output —
  caught early by the `blockiness_ratio` diagnostic.

---

## 4. Models

| model | what it is | parameters |
|---|---|---|
| **XGBoost, 2-stage** | stage 1 corrects IMERG at 10 km; stage 2 predicts the 1 km residual | ~220 trees, depth 4 |
| **CNN** | U-Net over the 1 km grid | 1.65 M |
| **Swin transformer** | SwinIR-style shifted-window attention | 7.9 M |
| **Diffusion** | conditional denoising diffusion over the residual from the CNN mean (CorrDiff-style); produces an *ensemble* | 12.4 M |

Every model outputs `bilinear(coarse) + correction` with a zero-initialised
head, so **an untrained model is exactly the bilinear baseline** and training
can only be measured as improvement over it.

---

## 5. Training protocol

```
train  2000-06-01 .. 2017-12-31   (6,423 days)
dev    2018                        (model selection, stacking weights)
test   2019-2020                   (731 days, scored once)
```

Splits are **temporal and identical across model families**. 2018 is a clean
hold-out for *every* model, because stacking weights are fitted there — a member
trained through 2018 would look artificially good and absorb all the weight.

* **Target transform**: `u = (log1p(p) − 0.497)/1.014`, fitted on training days.
* **Loss**: MSE in **mm space** (see §7 — this mattered a lot).
* **Sampling**: random 96 × 96 crops (8 × 8 coarse cells), wet-biased 0.8, with
  flip augmentation. Aspect is a direction, so a north–south flip sends θ→180−θ
  (cos flips sign) and east–west sends θ→−θ (sin flips sign).
* **Checkpoint selection**: `rmse_pod` — best RMSE *among checkpoints whose
  heavy-event POD clears bilinear's*. Selecting on RMSE alone can never keep an
  extremes-favouring model.
* **Optimiser**: AdamW, cosine decay, grad-clip 1.0, EMA 0.999, bf16 autocast
  with the **loss computed in fp32**.

---

## 6. Metrics, in one page

Full treatment in [METRICS.md](METRICS.md). Three families, because no single
number is sufficient:

**Accuracy** — RMSE, MAE, bias, Pearson r, NSE, KGE, median per-cell NSE.
**Realism** — spectral ratio below 10 km (1.0 = observed texture), FSS,
wet-area ratio, blockiness.
**Extremes / uncertainty** — POD and CSI above 30 mm, CRPS, ensemble spread.

**Two evaluation levels, never comparable.** Stage 1 is scored at 10 km against
block-averaged AORC; the final product at 1 km. Averaging 144 cells cancels
error, so a 10 km score always flatters. Compare only within a level.

**Why RMSE alone misleads.** The RMSE-optimal field is the conditional mean,
which for convective rain is a blur. Measured: XGBoost has the best RMSE
(4.655) and retains 1.9 % of observed sub-10 km variance; the diffusion member
has the worst RMSE (5.661) and retains 82 %. A displaced storm cell is punished
twice — missed where it was, false alarm where it was not — so squared error
actively prefers physically impossible fields. This is the *double penalty*.

---

## 7. Experiments, and what each one showed

| # | experiment | result |
|---|---|---|
| 1 | Two-stage tree vs bilinear (2015-18 train) | 1.3 % better RMSE; stage 2 **auto-rejected** |
| 2 | Loss space: log vs mm vs hybrid | mm-space MSE wins by **0.43 mm** on dev — biggest single fix |
| 3 | Capacity: 1.65 M vs 11.7 M CNN vs 7.9 M Swin | all within 0.005 RMSE → **information-limited, not capacity-limited** |
| 4 | Extremes: quantile loss, heavy weighting, importance weighting, uniform crops | only **heavy ×3 weighting** moved POD (0.612→0.633) at negligible RMSE cost |
| 5 | ERA5 predictors | 10 km 4.693 → 4.558; 1 km 4.856 → 4.750 |
| 6 | Training record 4 yr → 18.5 yr | 1 km 4.750 → 4.655; every deep model improved too |
| 7 | Diffusion ε- vs x₀- vs v-parameterisation | ε unusable (RMSE 23.2); x₀/v stable |
| 8 | min-SNR loss weighting | mean bias −1.95 → −0.28 mm; spread unchanged |
| 9 | Model stacking | equal-weight `xgb+cnn` 4.639 < best single 4.655 |
| 10 | Sub-daily structure features (tree) | 1 km 4.655 → **4.583**; `imerg_cond_intensity` ranks 4th of 31 |
| 11 | Deep models retrained on the same 31 features | CNN 4.694 → 4.663, Swin 4.729 → **4.653**, diffusion 4.840 → 4.807 |
| 12 | Stacking, after the tree gained structure features | blend 4.601 > XGBoost 4.583 → **no gain** |

### Negative results worth keeping

* **The 1 km residual stage has been rejected every single time** — with 4 years
  and with 18.5, with and without ERA5 (hold-out 5.524 vs bilinear 5.530). The
  pipeline decides this automatically via `upsampling.method: auto`. It is the
  strongest evidence that sub-grid placement is not recoverable from these
  inputs in this domain.
* **Architecture barely matters.** Four model families span 4.655–4.840.
* **Diffusion under-dispersion turned out to be informational, not
  algorithmic.** Neither the parameterisation change (ε → x₀ → v) nor min-SNR
  loss weighting moved the ensemble spread off ~0.57 of observed. Adding the
  sub-daily structure channels moved it to **0.81**. Learning a conditional
  *distribution* evidently needs more information than learning a conditional
  mean — the fix was better features, not a better sampler.
* **Stacking stopped helping once the members diverged.** An equal-weight
  `xgboost+cnn` blend beat the best single model while they were 0.04 apart;
  after structure features widened the gap to 0.11, the blend lost to XGBoost
  alone. Non-negative least squares had already signalled this by assigning
  Swin and diffusion zero weight.

---

## 8. Results (2019–2020 test period, 1 km vs AORC)

| product | RMSE | MAE | bias | NSE | KGE | CRPS | POD>30 | CSI>30 | spectral ratio |
|---|---|---|---|---|---|---|---|---|---|
| IMERG nearest | 4.994 | 1.529 | −0.080 | 0.647 | 0.787 | 1.529 | 0.600 | 0.440 | 32.7 |
| Bilinear | 4.919 | 1.506 | −0.082 | 0.657 | 0.787 | 1.506 | 0.601 | 0.445 | 0.125 |
| **XGBoost** | **4.583** | **1.467** | −0.046 | **0.702** | 0.739 | 1.467 | 0.548 | 0.458 | 0.021 |
| CNN | 4.663 | 1.511 | 0.044 | 0.692 | 0.784 | 1.511 | 0.614 | 0.468 | 0.162 |
| **Swin** | 4.653 | 1.520 | 0.145 | 0.693 | **0.790** | 1.520 | **0.633** | **0.469** | 0.109 |
| Diffusion (mean) | 4.807 | 1.493 | −0.113 | 0.673 | 0.746 | **1.050** | 0.511 | 0.413 | 0.248 |
| Diffusion (member) | 5.483 | 1.668 | −0.132 | 0.574 | 0.731 | 1.668 | 0.467 | 0.354 | **0.840** |

**Read it as a Pareto front, not a ranking.** Three products win different
things, and no product wins everything:

* **XGBoost** — accuracy: best RMSE, MAE, bias and NSE.
* **Swin** — extremes and variability: best POD, CSI and KGE, and the only
  product whose KGE exceeds bilinear's.
* **Diffusion** — probabilistic and realism: CRPS 1.050 (28 % better than any
  deterministic product) and a single member retaining 84 % of observed
  fine-scale variance.

Against the original plan's targets: 5.4 % RMSE reduction (target 20 %) and
POD 0.618 (target 0.80) — **both missed**. The domain is flat (8 of 151,200
cells exceed 5° slope) and IMERG is already nearly unbiased here, so there is
little recoverable 1 km signal. The targets assumed orographic structure this
domain does not have.

---

## 9. Bugs found — and how

Each of these would have silently corrupted a result.

| bug | symptom | how caught |
|---|---|---|
| Log-space loss, mm-space scoring | dev RMSE *rose* while training loss fell | watching both curves |
| Diffusion ε-parameterisation | `abar → 1e-8` at t=T amplifies error 10,000× | sampled spread 87 % too wide |
| `res_scale` from 8 fields | 40 % under-estimate, wrong SNR | measured on the true training distribution |
| CRPS counted NaN cells as zero | small bias in the headline probabilistic number | reading the mask logic |
| Stale merged NetCDF shadowing new output | RMSE disagreed with the validation log | cross-checking two sources |
| Stacking leakage | NNLS gave XGBoost 100 % of the weight | contradicted the test-set blend result |
| **Sync script shipping an explicit file list** | cluster ran 5-day-old code; results identical to previous run | byte-identical metrics |

The last one caused two wasted cluster runs. `scripts/vista_sync.sh` now uses
globs and **verifies per file after every push**.

---

## 10. Open gaps

1. **No independent validation.** AORC is an analysis, not truth — figure
   `24_fields_three_days.png` shows radar artifacts in AORC itself. A gauge
   comparison (GHCN-Daily) remains undone.
2. **Ensemble under-dispersion** (~57 % of observed spread).
3. **Daily only.** Sub-daily downscaling is harder and is where extremes matter.
4. **One domain.** A terrain-rich region would test whether the method works and
   Austin is simply hard.

---

## 11. Reproducing

```bash
conda env create -f environment.yml && conda activate downscale
./run_tests.sh                       # 30 tests, two processes (torch/OpenMP)

python main.py download               # NLCD, AORC, IMERG (needs Earthdata)
python main.py preprocess             # align all grids
python main.py features && python main.py train
python main.py predict && python main.py validate

python -m src.model_comparison        # scores every product together
python -m src.figures                 # and field_/explainer_/paper_figures
```

On the cluster: `scripts/vista_sync.sh push` then `sbatch slurm/<job>.slurm`.
GPU jobs need `module load gcc/13.2.0 cuda/12.5 python3/3.11.8` plus
`export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:$LD_LIBRARY_PATH` (the
module set ships a scipy built against a newer libstdc++ than it loads).
