# How this project is scored

## The task, stated precisely

IMERG arrives on a 0.1° grid (~10 km). AORC, the reference, is on a 1/120° grid
(~925 m). One coarse cell contains a **12 × 12 block of 144 fine cells**. Over the
Austin domain that is 1,050 coarse cells expanding to 151,200 fine cells, every
day, for 7,519 days.

So the model is asked to invent 143 numbers per coarse cell, constrained only by
their average. Whether that is possible is an empirical question, and most of
this metric suite exists to answer it honestly.

## Two evaluation levels — do not compare them

The pipeline has two stages and each is scored at its own resolution.

| stage | what it does | scored against | current RMSE |
|---|---|---|---|
| 1 — 10 km | corrects IMERG's errors at its native resolution | AORC block-averaged to 10 km | **4.476** |
| 2 — 1 km | produces the final 1 km field | AORC at native 1 km | **4.730** |

A 1 km score is always worse than a 10 km score on the same product, because
averaging 144 cells cancels independent error. **The two numbers are not
comparable**; only compare products *within* a level.

Baselines at 1 km: bilinear interpolation 4.919, nearest-neighbour 4.994. Any
learned product must beat bilinear or it has earned nothing — an untrained model
in this codebase literally *is* bilinear, because every network outputs
`bilinear(coarse) + correction` with a zero-initialised head.

## Glossary — every acronym, expanded

| acronym | full form | formula / meaning |
|---|---|---|
| **RMSE** | Root Mean Squared Error | `sqrt(mean((pred - obs)^2))` |
| **MAE** | Mean Absolute Error | `mean(abs(pred - obs))` |
| **bias** | mean error (not an acronym) | `mean(pred - obs)` |
| **r** | Pearson correlation coefficient | pattern agreement, magnitude-blind |
| **NSE** | Nash-Sutcliffe Efficiency | `1 - MSE/var(obs)`; 1 perfect, 0 = a constant, <0 worse |
| **KGE** | Kling-Gupta Efficiency | `1 - sqrt((r-1)^2 + (sd_p/sd_o - 1)^2 + (mu_p/mu_o - 1)^2)` |
| **CRPS** | Continuous Ranked Probability Score | `mean|X-y| - 0.5*mean|X-X'|`; equals MAE for a deterministic field |
| **POD** | Probability of Detection (hit rate) | `hits/(hits+misses)` |
| **FAR** | False Alarm Ratio | `false_alarms/(hits+false_alarms)` |
| **CSI** | Critical Success Index (threat score) | `hits/(hits+misses+false_alarms)` |
| **FSS** | Fractions Skill Score | `1 - mean((Pf-Po)^2)/(mean(Pf^2)+mean(Po^2))` over a neighbourhood |
| **RAPSD** | Radially Averaged Power Spectral Density | variance by spatial wavelength |
| **PSD** | Power Spectral Density | the spectrum RAPSD averages radially |

Non-acronym diagnostics: **spectral ratio <10 km** (predicted/observed power
below a 10 km wavelength), **wet-area ratio** (cells >= 1 mm, predicted/observed),
**blockiness ratio** (jump across coarse-block edges / jump inside; ~1 = no
artifacts), **median cell NSE**, **ensemble spread**.

Dataset and method acronyms used throughout: **IMERG** (Integrated
Multi-satellitE Retrievals for GPM), **GPM** (Global Precipitation Measurement),
**AORC** (Analysis of Record for Calibration), **NLCD** (National Land Cover
Database), **DEM** (Digital Elevation Model), **ERA5** (ECMWF Reanalysis v5),
**CAPE** (Convective Available Potential Energy), **TCWV**/**PWV** (Total Column
Water Vapour / Precipitable Water Vapour), **GHCN-D** (Global Historical
Climatology Network - Daily), **DDPM**/**DDIM** (Denoising Diffusion
Probabilistic / Implicit Models), **SNR** (signal-to-noise ratio), **NNLS**
(non-negative least squares), **EMA** (exponential moving average).

## The metrics, by what they ask

### Accuracy — is the amount right?

| metric | question it answers | watch out for |
|---|---|---|
| **RMSE** | typical error, squaring big misses | dominated by a few heavy days; rewards smoothing (see below) |
| **MAE** | typical error on an ordinary day | insensitive to extremes |
| **bias** | systematically too wet or too dry | `mean(pred) − mean(obs)`; can be near zero while every cell is wrong |
| **Pearson r** | is the spatial/temporal pattern right | ignores magnitude entirely |
| **NSE** | better than just predicting the long-run mean? | `1 − MSE/var(obs)`; 1 perfect, 0 worthless, negative = worse than a constant |
| **KGE** | right amount *and* right variability *and* right pattern | decomposes into r, variance ratio, bias ratio — unlike RMSE it **penalises over-smoothing** |
| **median cell NSE** | is skill uniform across the map | NSE per grid cell over time, then the median |

### Realism — does the field look like rain?

| metric | question it answers |
|---|---|
| **spectral ratio <10 km** | what fraction of the observed sub-10 km variance survives? 1.0 = realistic texture, 0.1 = 90 % of it lost, ≫1 = spurious noise or blocky artifacts. Also trainable — see *Scoring it is not the same as optimising it* below |
| **FSS** (Fractions Skill Score) | is rain in *approximately* the right place? Scored over a neighbourhood, so a small displacement is not fatal |
| **wet-area ratio** | is the right fraction of the map raining? |
| **blockiness ratio** | are there visible 10 km grid artifacts? ≈1 = none, ≫1 = the coarse grid is imprinted on the output |

### Extremes and uncertainty

| metric | question it answers |
|---|---|
| **POD > 30 mm** | of the cells where AORC really had ≥ 30 mm, what fraction did we flag? |
| **CSI** | hits, penalised for both misses and false alarms |
| **CRPS** | is the forecast both accurate *and* honestly uncertain? For a single deterministic field CRPS collapses exactly to MAE, so every product sits on one scale |
| **ensemble spread** | does the ensemble span the real uncertainty? Compared against observed residual spread; 0.59 means the ensemble is over-confident by ~40 % |

## Why RMSE alone is actively misleading here

This is the central methodological point of the project.

The information needed to place rain *within* a 12 × 12 block is largely not
present in the inputs. A 10 km rainfall average, a DEM and a land-cover map do
not say which square kilometre got the storm core. Given that, the field which
minimises mean squared error is the one that writes the block average into all
144 cells: **a blur**.

So RMSE actively rewards the least realistic product. The measured evidence:

| product | RMSE (1 km) | spectral ratio <10 km |
|---|---|---|
| XGBoost 2-stage | **4.583** (best) | **0.021** — retains 2 % of real texture |
| Bilinear | 4.919 | 0.125 |
| Diffusion, single member | 5.483 (worst) | **0.840** — realistic texture |

Of the products trained on a per-cell loss, the diffusion member is the only one
that looks like rainfall, and it scores worst on RMSE.

### The double penalty

Put a storm cell 3 km from where it actually fell and squared error punishes it
twice — a miss where the rain was, a false alarm where it was not. A smooth
field takes one moderate penalty everywhere and wins on RMSE while being
physically implausible. This is a well-known property of squared error on
intermittent fields, and it is why FSS and spectral diagnostics exist.

### The consequence for model selection

A single deterministic raster cannot be both RMSE-optimal and realistic. Those
are different estimators of different quantities:

* the **conditional mean** minimises squared error,
* a **sample from the conditional distribution** looks like rain,
* an **upper quantile** detects extremes.

Asking one field to be all three is a decision-theory error, not a
hyperparameter to tune. That is the argument for the diffusion model: it
produces the whole conditional distribution, from which the mean (for RMSE), a
member (for realism) and a quantile (for extremes) all follow.

It is also why the two-stage tree model keeps **auto-rejecting its own stage 2**
(hold-out 5.240 vs bilinear 5.241): the pipeline is reporting that no learnable
1 km signal exists here, rather than any human deciding so.

## Scoring it is not the same as optimising it

Everything above was for a long time a *reporting* convention: the spectral
ratio was measured, never trained. Every loss in the project was a per-cell
loss, and a per-cell loss is minimised by the conditional mean, which is smooth.
The trade-off the previous section describes is therefore partly a property of
the objective rather than of the problem.

`src/deep/spectral.py` adds a differentiable version of the same diagnostic —
square crop, demean, Hanning window, `|FFT2|²`, radial average, then the squared
gap between predicted and observed **log** power below the cut-off wavelength.
It reproduces the numpy scorer to a maximum relative difference of 7 × 10⁻¹⁶, so
the quantity trained is the quantity reported rather than a near relative.

    python -m src.deep.train --model cnn --loss mse --spectral-weight 0.01 --spectral-cut 10

It is **additive** to any `--loss`, because structure is orthogonal to the
intensity space the loss is taken in. `--spectral-weight 0` is byte-identical to
the previous objective. Dev evaluation gained a `spec` column so the effect is
visible during training.

Measured effect, matched control against `w = 0.01`:

| input | RMSE | spectral ratio | POD > 30 mm |
|---|---|---|---|
| IMERG 10 km, control | 4.654 | 0.303 | 0.540 |
| IMERG 10 km, + spectral | **4.635** | **1.260** | 0.533 |
| POWER 50 km, control | **5.782** | 0.167 | 0.391 |
| POWER 50 km, + spectral | 5.821 | **1.281** | **0.414** |

Three caveats worth carrying. It **overshoots** — both land near 1.3 rather than
1.0, so over-smoothing is traded for mild over-roughening. The spectral ratio is
an **amplitude** diagnostic and cannot distinguish correct structure from
well-sized noise; FSS can, and it improves at every neighbourhood width for the
1 mm threshold, which noise would have degraded. And the checkpoint is selected
on RMSE alone, so on POWER — where texture decays through training — the result
depends on the RMSE optimum falling early. There is no spectral analogue of the
`rmse_pod` floor yet.

See §4.8 of [RESEARCH_PAPER.md](RESEARCH_PAPER.md) for the weight sweep, the
three-seed test and the full FSS breakdown.

## How to read the comparison table

1. Check **RMSE and bias** — is the magnitude right?
2. Check **KGE and the spectral ratio** — or the RMSE winner may simply be the blurriest.
   Note whether the product was trained with a spectral term; if it was, the ratio is an
   optimised quantity and no longer an independent check on it.
3. Check **POD > 30 mm** against bilinear — learned models routinely lose here.
4. For the ensemble, check **CRPS and spread ratio** together — good CRPS with 0.59 spread is an over-confident forecast.

A product is only genuinely better if it wins in more than one family.

## Caveat on the reference

AORC is a gridded analysis, not truth. Figure `24_fields_three_days.png` shows
radial spokes and banding in AORC itself on low-intensity days — radar artifacts
in the reference. That observation now has a documented cause: AORC's primary
daily CONUS source has been Stage IV radar QPE since 2002, and **since January
2016 it receives no bias adjustment at all** (Fall et al., 2023). Our test
period, 2019–2020, is entirely inside that unadjusted era, so every score here
is against a radar analysis rather than a gauge-constrained one.

The methodology also changes *inside* our record, with training almost entirely
in the adjusted era and dev and test outside it. `src/reference_drift.py`
measures the reference against itself year by year; no era-to-era step in mean
rainfall, wet-area fraction, 99th percentile or spectral shape exceeds one
standard deviation of interannual variability, so we keep the splits. See §2.5
of [RESEARCH_PAPER.md](RESEARCH_PAPER.md).

Every number here is *relative to AORC*, and the GHCN-D gauge comparison of §4.3
is the only partly independent check.
