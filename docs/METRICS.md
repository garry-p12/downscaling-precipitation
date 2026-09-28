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
| **spectral ratio <10 km** | what fraction of the observed sub-10 km variance survives? 1.0 = realistic texture, 0.1 = 90 % of it lost, ≫1 = spurious noise or blocky artifacts |
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
| XGBoost 2-stage | **4.750** (best) | **0.020** — retains 2 % of real texture |
| Bilinear | 4.919 | 0.125 |
| Diffusion, single member | 5.945 (worst) | **0.905** — realistic texture |

The diffusion member is the only product that looks like rainfall, and it scores
worst on RMSE.

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

## How to read the comparison table

1. Check **RMSE and bias** — is the magnitude right?
2. Check **KGE and the spectral ratio** — or the RMSE winner may simply be the blurriest.
3. Check **POD > 30 mm** against bilinear — learned models routinely lose here.
4. For the ensemble, check **CRPS and spread ratio** together — good CRPS with 0.59 spread is an over-confident forecast.

A product is only genuinely better if it wins in more than one family.

## Caveat on the reference

AORC is a gridded analysis, not truth. Figure `24_fields_three_days.png` shows
radial spokes and banding in AORC itself on low-intensity days — radar artifacts
in the reference. Every number here is *relative to AORC*, and an independent
gauge comparison remains an open gap.
