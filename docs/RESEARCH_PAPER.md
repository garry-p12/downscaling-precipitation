# Orographic forcing, not model capacity, sets the ceiling on satellite precipitation downscaling

### A controlled two-domain study: 10 km → 1 km over flat central Texas and the Colorado Front Range, with a 50 km reanalysis input as a third arm, four model families, 18.5 years of record, seed repeats, and a spectral training objective

---

## Abstract

Satellite precipitation retrievals such as IMERG are delivered at ~10 km, while flood
modelling, small-catchment hydrology and urban applications require ~1 km. We downscale
IMERG V07 daily precipitation from 0.1° to 1/120° over **two domains of identical size**
— 1,050 coarse cells and 151,200 fine cells each, same 18.5-year record, same temporal
splits, same 31 predictors — differing only in terrain regime. The first is centred on
Austin, Texas, where **0.000** of cells exceed 5° slope. The second spans the Colorado
Front Range, where **25.5 %** do and elevation runs 1,221–4,245 m. AORC is the 1 km
reference; evaluation is on an unseen 2019–2020 test period (731 days).

Skill differs by more than a factor of two: **6.8 % RMSE reduction against bilinear
interpolation in Texas, 15.0 % in Colorado.** Decomposing Colorado by season separates
the mechanism from the geography. The cool half-year, when terrain rather than
convection decides where precipitation lands, reaches **20.8 %** — meeting the 20 %
target that is missed everywhere else. The warm half-year, when mountain precipitation
is convective, reaches **8.0 %**, almost exactly the flat-Texas figure of 6.8 %.
Downscaling skill therefore tracks how orographically *forced* the precipitation is,
not how much terrain the domain contains.

Within each domain, architecture is irrelevant. Four model families spanning
gradient-boosted trees to a 12.4 M-parameter diffusion model converge to within 4.9 %
of each other, and seed repeats bound run-to-run noise at **0.15 %** for trees and
**~0.8 %** for both deep models. Those repeats also show a single training run to be
unreliable in both directions: the published CNN–Swin gap of 0.010 mm day⁻¹ understates
a real difference of 0.040 that holds across all nine pairwise seed comparisons. What moved the numbers was information: ERA5 predictors
(−2.2 %), a 4.6× longer record (−2.0 %), and four sub-daily structure features obtained
at no extra download cost from counters already present in the daily granule (−1.5 %).

Seven attempts to recover the genuine sub-grid component failed. A 1 km residual stage
was auto-rejected on every configuration in both domains, including the orographic
season where terrain features rank first and second in its importances and the gain is
consistently positive but only 0.2–0.4 %. We conclude that bias correction and
neighbourhood structure at the coarse scale are recoverable, while sub-grid *placement*
largely is not — and that the difference between domains is how much of the first a
forcing mechanism supplies.

Accuracy and realism appear to be distinct estimators that cannot be jointly maximised:
the best-RMSE product retains **2 %** of observed sub-10 km variance, and the only
product with realistic texture — a single diffusion member — has the worst RMSE of all.
We show this is partly an artefact of the *objective* rather than of the problem. Adding
a differentiable spectral penalty to the training loss raises retained variance from
0.30 to **1.26** on the test set while leaving RMSE unchanged within seed noise
(paired three-seed difference **+0.004 ± 0.021 mm day⁻¹**). Fractions-skill analysis
confirms the added variance is structure and not well-sized noise, though it is
light-rain structure on the 10 km input, though on a 50 km input heavy-event detection
improves as well.

**Keywords**: statistical downscaling, IMERG, AORC, orographic precipitation, diffusion
models, double penalty, spectral fidelity, negative results

---

## 1. Introduction

### 1.1 The problem is under-determined by construction

NASA's IMERG provides globally gridded precipitation at 0.1° (~10 km). One coarse cell
contains a 12 × 12 block of 144 one-kilometre cells. A downscaler must produce 143
numbers per cell that the input does not determine, constrained only by their mean.

This is not an interpolation problem with a better interpolator waiting to be found. It
is an inference problem whose answer depends on whether the sub-grid pattern is *forced*
by something observable. Two regimes exist:

- **Orographically forced.** Terrain fixes where the rain falls. Moist air is lifted
  over a slope, cools, and condenses; the windward side is wet and the lee is dry, every
  time. The mapping from (coarse total, elevation, slope, wind direction) to the fine
  field is close to deterministic, and the literature's 15–25 % RMSE reductions come
  from here.
- **Convectively forced.** The sub-grid pattern is set by where individual storm cells
  happen to organise. It is chaotic on the scales that matter and absent from every
  coarse predictor.

Most published downscaling studies report results from a single domain, which conflates
method with regime. We instead hold everything constant except the regime.

### 1.2 Contributions

1. **A controlled two-domain comparison.** Two boxes of *identical* size (1,050 coarse
   cells, 151,200 fine cells), identical period, identical splits, identical 31-channel
   predictor set, differing only in terrain: Austin at 0.000 fraction of cells above 5°
   slope, Colorado at 0.255. Skill differs by 2.2× (6.8 % vs 15.0 %), isolating regime
   as the cause (§4.5).

2. **A seasonal decomposition that identifies the mechanism.** Within the terrain-rich
   domain, cool-season skill (20.8 %) exceeds warm-season skill (8.0 %) by a factor of
   2.6, and the warm-season figure matches the flat domain's all-year figure. Terrain
   alone does not create skill; orographic *forcing* does (§4.6).

3. **A free, high-value predictor source.** The IMERG *daily* granule carries half-hour
   counters (`precipitation_cnt`, `precipitation_cnt_cond`) that recover conditional
   rain intensity — convective vs stratiform character — without downloading the
   half-hourly product (360,912 granules, ~33 h). Cost: 62 KB/day instead of 33 KB, the
   same number of requests (§3.2, §5.2).

4. **Seed repeats that bound the noise floor.** Four XGBoost seeds span 0.15 %; three
   CNN seeds span 0.8 %. This retires one comparison the single-run results appeared to
   support, and establishes which margins are interpretable (§5.8).

5. **A systematic negative result on architecture.** Across 1.65 M / 7.9 M / 11.7 M /
   12.4 M parameter models and gradient-boosted trees, dev RMSE spans 0.05 %. Capacity
   and inductive bias are not the binding constraint (§5.1).

6. **An evaluation protocol that exposes the accuracy–realism trade-off**, combining
   spectral ratio below 10 km, the Fractions Skill Score, and CRPS placed on a common
   scale with deterministic MAE, plus independent validation against 616 rain gauges.

7. **An explicit statement of effective sample size.** 151,200 cells × 6,423 days is not
   970 M independent samples; cells within a day are strongly correlated and the
   independent unit is the weather day (~6,400, of which ~1,100 are meaningfully wet).

8. **A spectral training objective that partly dissolves the accuracy–realism
   trade-off.** A differentiable penalty on the log power spectrum below the native
   resolution, implemented to agree with the scoring diagnostic to machine precision,
   recovers 4.2× the fine-scale variance at no RMSE cost detectable against a
   three-seed noise floor (§4.8). It is additive to any per-cell loss and costs nothing
   measurable in throughput.

---

## 2. Data

### 2.1 Sources

| dataset | role | native resolution | access |
|---|---|---|---|
| **IMERG** `GPM_3IMERGDF` V07 Final | predictor | 0.1°, daily | NASA GES DISC, OPeNDAP server-side subsetting |
| **AORC** v1.1 | target and reference | 1/120°, hourly → daily | NOAA on AWS S3 (anonymous) |
| **ERA5** | predictor | 0.25°, hourly → daily | ARCO-ERA5 on GCS (anonymous) |
| **NLCD 2021** | predictor | 30 m → 1 km | USGS MRLC WCS |
| **Copernicus GLO-90 DEM** | predictor | 90 m → 1 km | AWS (anonymous) |
| **GHCN-Daily** | independent validation | point gauges | NOAA NCEI |

`randomError` is available in the V07 daily granule but was excluded: it is labelled
mm day⁻¹ yet carries a median near 5,000 mm day⁻¹, so it cannot be used as an
uncertainty estimate.

### 2.2 Grid construction

The coarse grid is IMERG's native 0.1° grid, cell centres at `edge + 0.05`. The fine
grid is AORC's native 1/120° grid, cell centres at exact multiples of 1/120°. The
refinement factor is 12, giving 144 fine cells per coarse cell.

**The fine grid *is* AORC's grid**, so the 1 km reference is never interpolated. The
cost is that the 12 × 12 block centre sits 460 m from the IMERG cell centre — under 5 %
of a coarse cell width, and preferable to resampling the target.

### 2.3 The two domains

Both boxes were sized to **exactly 1,050 coarse cells and 151,200 fine cells** so that
sample size, cell count and refinement factor cannot explain any difference between them.

| | **Austin, TX** | **Colorado Front Range** |
|---|---|---|
| bounding box | `[-99.0, 28.5, -96.0, 32.0]` | `[-107.0, 38.5, -104.0, 42.0]` |
| coarse / fine cells | 1,050 / 151,200 | 1,050 / 151,200 |
| elevation range | 100–668 m | **1,221–4,245 m** |
| mean slope | — | 3.50° (max 31.41°) |
| **cells above 5° slope** | **0.000** | **0.255** |
| training rows | ~6.70 M | 6,743,415 |
| cells ≥ 30 mm day⁻¹ (test) | **2.02 %** | **0.30 %** |
| raw IMERG correlation vs AORC | 0.821 | 0.573 |

The Colorado box spans the Continental Divide (Longs Peak 4,346 m, Mt Evans 4,348 m),
the Denver–Boulder–Fort Collins corridor, and high plains to the east, giving a strong
west-to-east orographic gradient *inside* the evaluation box.

Two asymmetries are worth stating up front because they complicate naive comparison.
Colorado is **drier**, so its absolute RMSE is lower and only the *percentage* reduction
is comparable across domains. And IMERG performs far worse there to begin with
(correlation 0.573 against 0.821), which is itself diagnostic: retrieval over complex,
frequently snow-covered terrain is harder.

### 2.4 Temporal splits

IMERG Final V07 daily begins 2000-06-01, which is the binding constraint; AORC and ERA5
both start earlier. Splits are identical for both domains and all model families.

```
train  2000-06-01 .. 2017-12-31    6,423 days
dev    2018-01-01 .. 2018-12-31      365 days   model selection, stacking weights
test   2019-01-01 .. 2020-12-31      731 days   scored once
```

Splits are temporal and identical across all model families, including the tree models.
This matters specifically because the stacking weights are fitted on the dev year: a
member trained through 2018 appears artificially strong and absorbs all the weight. An
earlier configuration with `train_end: 2018-12-31` produced exactly this failure, with
NNLS assigning the tree 100 % — caught by an explicit leakage check that refuses to fit
weights on a period any member saw in training.

### 2.5 Provenance of the reference, and a methodology break inside our splits

AORC is the target for every number in this paper, so how it is built matters.
It does not use one precipitation methodology throughout. From Fall et al. (2023):

| period | primary daily CONUS source | adjustment |
|---|---|---|
| 1979–2001 | NLDAS-2 (CPC gauge analysis, temporally disaggregated) | strong Livneh monthly constraint |
| 2002–2009 | **Stage IV radar QPE** | strong Livneh monthly constraint |
| 2010–2015 | Stage IV radar QPE | climatological constraint only |
| **2016–present** | Stage IV radar QPE, CPC as secondary | **none** |

Two consequences follow, and the first is a hazard for this study specifically.
**Our training period is almost entirely inside the adjusted era while dev and
test are entirely outside it.** If the unadjusted product carries different
fine-scale structure, the sub-10 km variance a model is trained toward is not
the variance it is scored against, which would bias §4.8's texture results
independently of any model or loss.

We tested the reference against itself, year by year over the evaluation box,
on the 40 wettest days of each year (`src/reference_drift.py`):

| | adjusted, 2000–2015 | unadjusted, 2016–2020 | step, in units of the within-era interannual SD |
|---|---|---|---|
| domain-mean rainfall | 2.438 mm day⁻¹ | 2.699 | 0.37 |
| wet-area fraction | 0.794 | 0.834 | 0.46 |
| 99th percentile | 88.9 mm | 103.6 | 0.78 |
| share of variance below 10 km | 7.0 × 10⁻⁵ | 5.3 × 10⁻⁵ | 0.74 |

**No step exceeds one standard deviation of year-to-year variability**, which
itself spans a factor of three in domain-mean rainfall (1.16 in 2011 to 3.65 in
2015). The test years' texture share (6 × 10⁻⁵ in 2019, 4 × 10⁻⁵ in 2020) sits
in the middle of the training distribution. We therefore keep the splits: there
is no measurable discontinuity to correct for, and restricting the record to a
single methodological era would cost most of the 18.5 years that §5.2 shows are
worth 2.0 % of skill.

The one residual signal is the 99th percentile, which rises monotonically across
the four eras (80.2 → 87.3 → 93.9 → 103.6 mm). That ordering is what progressive
loosening of a climatological constraint would produce, but it is confounded by
individual events — 2015 and 2017 are the two largest years and both sit in
earlier eras — so we do not claim it as a methodology effect. It does mean the
heavy-event statistics of §4.1 are measured against a reference whose extremes
are larger than those of most training years.

The second consequence is interpretive. Our test period is **unadjusted Stage IV
radar QPE**, and this explains an artefact we had noted empirically but not
accounted for: AORC itself shows radial spokes and banding on low-intensity days
(figure `24_fields_three_days`). Those are radar artefacts in the reference, now
with a documented cause. Any product scored against AORC after 2016 is being
compared with a radar analysis, not with a gauge-constrained one.


---

## 3. Methods

### 3.1 Predictor set

**31 conditioning channels at 10 km**, in five groups:

| group | features | rationale |
|---|---|---|
| IMERG dynamic | `imerg`, `imerg_roll7`, `imerg_pct`, `imerg_nbr3` | the retrieval, its 7-day history, its rank against local climatology, its 3 × 3 neighbourhood |
| **Sub-daily structure** | `imerg_wet_frac`, `imerg_cond_intensity`, `imerg_mw_frac`, `imerg_prob_liquid` | §3.2 |
| ERA5 atmosphere | `cape`, `cape_max`, `tcwv`, `tcwv_max`, `t2m`, `wind_speed`, `wind_sin`, `wind_cos`, `moist_flux` | the storm environment the retrieval cannot see |
| Static surface | `dem_mean`, `dem_std`, `slope_mean`, `imperv_mean`, 8 × `lc_frac_*` | terrain and land cover |
| Temporal | `doy_sin`, `doy_cos` | seasonality |

A further **23 features at 1 km** serve the optional residual stage: bilinearly upsampled
coarse fields (`imerg_bl`, `pred_bl`), 1 km terrain and land cover, and interactions
(`dem_x_slope`, `dem_anom`).

`imerg_pct` deserves note: it converts "12 mm" into "a 97th-percentile day *at this
location*", allowing a single model to span a wetter east and drier west without
per-cell parameters.

### 3.2 Sub-daily structure features

A daily total cannot distinguish 38 mm delivered in six violent half-hours from 38 mm
spread across thirty gentle ones, yet these produce entirely different sub-grid fields —
the first concentrates rain into a few square kilometres, the second spreads it evenly.

The direct route to this information is the half-hourly product: 48 granules × 7,519
days = **360,912 granules**, roughly 33 hours of transfer. Instead, the daily granule's
own counters give:

| feature | definition | captures |
|---|---|---|
| `imerg_cond_intensity` | total ÷ (wet half-hours × 0.5 h) | mm h⁻¹ *while raining* — convective vs stratiform |
| `imerg_wet_frac` | wet ÷ valid half-hours | what fraction of the day was wet |
| `imerg_mw_frac` | microwave ÷ merged estimate | genuine overpass vs IR morphing — retrieval trust |
| `imerg_prob_liquid` | phase probability | rain vs frozen |

Marginal cost is one longer OPeNDAP query string.

### 3.3 Model families

All models are parameterised as `bilinear(coarse) + correction` with a zero-initialised
output head, so **an untrained model is exactly the bilinear baseline**. Every reported
gain is therefore measured against interpolation by construction, not by comparison.

| model | description | size |
|---|---|---|
| **XGBoost, 2-stage** | stage 1 corrects IMERG at 10 km; stage 2 optionally predicts the 1 km residual | depth 4, lr 0.02, ≤2000 trees with early stopping |
| **CNN** | U-Net over the 1 km grid | 1.65 M (base 24); 11.7 M variant (base 64) for the capacity ablation |
| **Swin transformer** | SwinIR-style shifted-window attention, 4 RSTB groups × 6 layers, dim 180, 6 heads | 7.9 M |
| **Residual diffusion** | conditional DDPM over the residual from a frozen CNN mean (CorrDiff-style), cosine schedule, v-parameterisation, 40-step DDIM sampling, 6 members | 12.4 M |

Stage 2 is gated: `upsampling.method: auto` adopts the residual model only if it beats
bilinear by more than 0.5 % on a training-tail hold-out. The gate can be restricted to a
season via `model.fine.months`, which §5.5 uses.

### 3.4 Training protocol

Patch size 96, batch 32, AdamW at 2 × 10⁻⁴ with 500-step warmup and cosine decay, EMA
0.999, bf16 mixed precision, with early selection. Loss is mm-space masked MSE (§5.3),
optionally with the additive spectral penalty of §4.8. Checkpoint selection uses
`rmse_pod`: minimum dev RMSE subject to dev POD ≥ 0.633, which prevents selecting an
over-smoothed checkpoint that wins RMSE by refusing to predict extremes.

The POD floor is domain-specific and silently disables selection if it is not reset. It
is bilinear's dev POD on the Austin domain; on POWER, where bilinear reaches only 0.288,
no checkpoint ever clears 0.633, every evaluation is ineligible, and the end-of-run
fallback ships the *last* step rather than the best one. A POWER Swin run lost 1.7 % of
dev RMSE this way before the floor was reset to 0.29. We report it because the failure is
silent: the training log prints a plausible "best dev score" either way.

Deep models trained on NVIDIA GH200 (TACC Vista) and A100 (TACC Lonestar6); the tree
pipeline is CPU-only. Seed repeats (§5.8) match the **step budget** of the original run
rather than its wall-clock, because the published Swin run was time-limited at 12,494
steps and matching minutes across different GPUs would confound hardware with seed.

### 3.5 Evaluation protocol

Metrics fall into three families because no single number is sufficient. Full
definitions are in [METRICS.md](METRICS.md).

**Accuracy** — RMSE, MAE, bias, Pearson *r*, NSE, KGE. KGE is reported alongside RMSE
specifically because it decomposes into correlation, variability ratio and bias ratio,
and therefore *penalises* the over-smoothing that RMSE *rewards*.

**Realism** — spectral ratio below 10 km wavelength (predicted ÷ observed variance;
1.0 is correct and the diagnostic is two-sided, since smoothing loses power and noise
adds too much), FSS at multiple neighbourhood widths, wet-area ratio, and a blockiness
ratio that detects imprinted coarse-grid structure.

**Extremes and uncertainty** — POD, FAR and CSI above 30 mm day⁻¹; CRPS, which reduces
exactly to MAE for a deterministic field and therefore places ensembles and single
rasters on one scale.

**Two evaluation levels are never compared.** The same prediction scores far better at
10 km than at 1 km purely because averaging 144 cells cancels independent error. All
headline numbers are at 1 km against AORC.

---

## 4. Results

### 4.1 Headline scores, Austin (flat)

Test period 2019-01-01 to 2020-12-31, 731 days, 1 km, against AORC.

| product | RMSE | MAE | bias | *r* | NSE | KGE | CRPS | POD>30 | CSI>30 | FSS 10 mm@15 km | spectral ratio <10 km | wet-area ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IMERG nearest | 4.994 | 1.529 | −0.080 | 0.810 | 0.647 | 0.787 | 1.529 | 0.600 | 0.440 | 0.759 | 32.71 | 0.997 |
| Bilinear | 4.919 | 1.506 | −0.082 | 0.815 | 0.657 | 0.787 | 1.506 | 0.601 | 0.445 | 0.757 | 0.125 | 1.011 |
| **XGBoost 2-stage** | **4.583** | **1.467** | **−0.046** | **0.839** | **0.702** | 0.739 | 1.467 | 0.548 | 0.458 | **0.769** | 0.021 | 1.320 |
| CNN (U-Net) | 4.663 | 1.511 | 0.044 | 0.832 | 0.692 | 0.784 | 1.511 | 0.614 | 0.468 | 0.760 | 0.162 | 1.105 |
| **Swin transformer** | 4.653 | 1.520 | 0.145 | 0.834 | 0.693 | **0.790** | 1.520 | **0.633** | **0.469** | 0.764 | 0.109 | 1.235 |
| Diffusion (ens. mean) | 4.807 | 1.493 | −0.113 | 0.820 | 0.673 | 0.746 | **1.050** | 0.511 | 0.413 | 0.762 | 0.248 | 1.159 |
| Diffusion (1 member) | 5.483 | 1.668 | −0.132 | 0.766 | 0.574 | 0.731 | 1.668 | 0.467 | 0.354 | 0.740 | **0.840** | 1.105 |

**No product wins everything, and the three that win something win different things.**

- **XGBoost** takes accuracy: best RMSE (6.8 % below bilinear), MAE, bias, NSE, *r*, FSS.
- **Swin** takes extremes and variability: best POD, CSI, and the only KGE exceeding
  bilinear's — it is the only learned product that does not lose variability.
- **Diffusion** takes the probabilistic and realism axes: CRPS 1.050 against 1.467 for
  the best deterministic product (28 % better), and a single member retaining 84 % of
  observed sub-10 km variance.

**Reporting POD alone misrepresents heavy-rain skill.** The detection suite at
30 mm day⁻¹, which the pipeline has always computed and this paper had not shown:

| product | POD | FAR | CSI | frequency bias | FSS at 1 / 5 / 15 / 41 cells |
|---|---|---|---|---|---|
| IMERG nearest | 0.600 | 0.377 | 0.440 | 0.964 | 0.612 / 0.639 / 0.688 / 0.760 |
| Bilinear | 0.601 | 0.368 | 0.445 | 0.952 | 0.616 / 0.641 / 0.686 / 0.756 |
| XGBoost 2-stage | 0.548 | **0.262** | 0.458 | 0.742 | 0.629 / 0.654 / 0.697 / 0.767 |
| CNN (U-Net) | 0.614 | 0.338 | 0.468 | 0.927 | 0.637 / 0.663 / 0.706 / 0.773 |
| **Swin transformer** | **0.633** | 0.355 | **0.469** | 0.981 | **0.639 / 0.665 / 0.708 / 0.776** |
| Diffusion (ens. mean) | 0.511 | 0.317 | 0.413 | 0.749 | 0.585 / 0.612 / 0.658 / 0.734 |
| Diffusion (1 member) | 0.467 | 0.407 | 0.354 | 0.788 | 0.523 / 0.555 / 0.611 / 0.707 |

XGBoost's low POD is usually read as a failure to detect extremes. It is accompanied by
a **false-alarm ratio of 0.262 against bilinear's 0.368**, and a higher CSI. Every
learned product beats bilinear on the neighbourhood score at every width. A pointwise
POD rewards over-forecasting, and frequency bias shows which products do it: XGBoost
forecasts heavy rain 0.742 times as often as it occurs, the signature of a conditional
mean, while Swin is nearly unbiased at 0.981.

Neighbourhood FSS matters here for the same reason the double penalty does (§4.2): a
correctly sharp storm core displaced by a few kilometres is scored as a miss *and* a
false alarm pointwise, while a hydrologist would accept it. That is also why a single
diffusion member has the best texture in the study and nearly the worst POD.

Note the CNN and Swin RMSE values differ by only 0.010 mm day⁻¹ here. §5.8 shows that
single pairing is unrepresentative: across three seeds each, Swin leads by **0.040 mm
day⁻¹** with disjoint ranges. This table understates a real difference rather than
inventing one.

### 4.2 The accuracy–realism trade-off

RMSE alone is actively misleading here. The field minimising squared error is the
conditional mean, which for convective rainfall is a blur. A correctly-sharp field
displaced by 8 km is penalised twice — a miss where the rain was and a false alarm where
it was not — and scores roughly 1.7× worse than a smooth field carrying the same total
water at 20 % of the true peak.

The empirical consequence is visible in the table: the products with the best RMSE
retain **2–16 %** of observed fine-scale variance, and the only product with realistic
texture has the **worst** RMSE. This is not a tuning failure. The conditional mean, a
draw from the conditional distribution, and an upper quantile are three different
estimators serving three different decisions. Requiring one raster to be all three is a
decision-theory error, and it is the principal argument for a generative product. §4.8
qualifies this: part of the trade-off is a property of the per-cell objective rather than
of the problem.

![Accuracy against realism](../results/figures/publication/F3_accuracy_realism_plane.png)

*RMSE against retained sub-10 km variance, for both coarse inputs. The vertical axis is
logarithmic and the dashed line marks correct texture. Products on the left are accurate,
products near the line are realistic, and until §4.8 nothing was both.*

![Power spectra](../results/figures/publication/F4_power_spectra.png)

*Radially averaged power spectral density against AORC. The shaded band is the sub-10 km
range the spectral ratio summarises. Bilinear and XGBoost fall away by an order of
magnitude or more inside it; a single diffusion member tracks the observations, and the
spectral-penalty CNN (dashed) does so from a deterministic model.*

### 4.3 Independent validation against rain gauges

616 GHCN-Daily stations with ≥500 valid days in the test period, 412,683 station-days.
GHCN daily totals end at a local observation hour (modal 0700), so a per-station day
offset was fitted: 520 stations align at shift 0, 93 at −1 day, 3 at +1.

| product | RMSE | MAE | bias |
|---|---|---|---|
| AORC (the reference itself) | 6.314 | 1.831 | +0.017 |
| **XGBoost** | **6.754** | **2.235** | **+0.004** |
| Diffusion (ens. mean) | 6.888 | 2.241 | −0.104 |
| Stacked blend | 6.848 | 2.264 | +0.019 |
| CNN | 6.993 | 2.323 | +0.091 |
| Swin | 7.032 | 2.342 | +0.175 |
| Bilinear | 7.220 | 2.317 | −0.021 |

**The ranking survives against instruments.** XGBoost is 6.5 % better than bilinear here
versus 6.8 % against AORC, so the gain is a property of the product and not an artefact
of fitting the reference. Absolute values are ~40 % higher than against AORC because a
gauge is a point and a cell is ≈0.74 km²; only the ordering is interpretable. AORC's own
lead is expected, since it assimilates gauge data — this validation is therefore only
partly independent, and is reported as a consistency check rather than as ground truth.

### 4.4 Ensemble calibration

Six members, 40 DDIM steps: CRPS 1.050 mm day⁻¹, mean ensemble spread 0.995 mm day⁻¹.
Spread reached 0.81 × observed after the sub-daily features were added, from ~0.57
before (§5.2). The ensemble remains mildly under-dispersed, which is the expected
direction for a residual diffusion model trained on a limited number of independent
weather days.

### 4.5 The terrain-rich domain: skill more than doubles

Colorado Front Range, same 731 test days, same 31 features, same 1,050 coarse cells.
Tree pipeline only; deep models were not retrained on this domain.

| metric | **Colorado ML** | Colorado bilinear | Colorado nearest | **Austin ML** | Austin bilinear |
|---|---|---|---|---|---|
| **RMSE** | **2.825** | 3.324 | 3.360 | 4.583 | 4.919 |
| **reduction vs bilinear** | **15.0 %** | — | — | **6.8 %** | — |
| MAE | 1.167 | 1.306 | 1.322 | 1.467 | 1.506 |
| bias | **−0.244** | −0.687 | −0.688 | −0.046 | −0.082 |
| Pearson *r* | **0.717** | 0.576 | 0.561 | 0.839 | 0.815 |
| NSE | **0.493** | 0.299 | 0.284 | 0.702 | 0.657 |
| **KGE** | **0.474** | 0.215 | 0.216 | 0.739 | **0.787** |
| median cell NSE | 0.490 | 0.303 | 0.283 | 0.711 | 0.661 |
| frac. cells NSE > 0.6 | 0.036 | — | — | 0.894 | — |
| POD > 30 mm | 0.051 | 0.034 | 0.037 | 0.548 | 0.601 |
| blockiness ratio | 1.006 | 1.005 | 999.0 | 1.002 | 0.998 |

![Two domains compared](../results/figures/publication/F2_two_domains.png)

*Open circles are bilinear, filled are the ML product, on identical axes. In Colorado
the model improves every metric shown; in Austin it loses KGE.*

Four observations, in decreasing order of importance.

**The margin over interpolation more than doubles: 15.0 % against 6.8 %.** Because the
two domains are identical in size, period, splits and predictors, regime is the only
variable that differs.

**KGE flips sign relative to bilinear.** In Austin the ML product *loses* on KGE (0.739
against 0.787) — it buys RMSE partly by damping variability. In Colorado it wins
decisively (0.474 against 0.215). The metric that punishes over-smoothing now favours
the model, which is a qualitatively different result from the flat domain.

**Terrain finally enters the shipped model.** Stage-1 importances:

```
imerg_nbr3, imerg, imerg_wet_frac, t2m, cape, imerg_pct,
imerg_prob_liquid, imerg_cond_intensity, dem_mean, dem_std
```

`dem_mean` 9th and `dem_std` 10th. In Austin `dem_mean` ranked 19th of 31 and no terrain
feature reached the top ten. Note also `imerg_prob_liquid` at 7th — the feature that
ranked **dead last (31/31)** in Texas, where rainfall is almost always liquid. Its rise
is an independent confirmation that this is a different precipitation regime, and that
phase discrimination matters where snow does.

**Per-cell skill is nonetheless worse.** Median cell NSE 0.490 against Austin's 0.711,
and only **3.6 %** of cells exceed NSE 0.6 against Austin's 89.4 %. The domain-wide gain
is large while the typical individual location is harder to predict — mountain
precipitation is more spatially variable, and IMERG starts from a much weaker retrieval
(correlation 0.573). The two statements are compatible: the model removes a large
systematic error (bias −0.687 → −0.244) without making any given square kilometre easy.

**Heavy-event detection fails in Colorado.** POD above 30 mm day⁻¹ is 0.051 for ML and
0.034 for bilinear — neither has meaningful skill. This is *not* a rarity artefact: the
Colorado test period contains 335,936 cells above 30 mm (0.30 % of all cells). Austin
contains 2,236,099 (2.02 %), so Austin actually has 6.6× more heavy cells and achieves
POD 0.548. Mountain extremes are genuinely harder, and both the retrieval and the
downscaler miss them.

### 4.6 Seasonal decomposition: forcing, not terrain, is the variable

Splitting the Colorado test period into a cool half-year (October–April) and a warm
half-year (May–September) separates orographic from convective forcing within a single
domain, holding geography fixed.

| regime | ML RMSE | bilinear RMSE | **reduction** |
|---|---|---|---|
| **Colorado, cool season (Oct–Apr)** — orographic | 2.599 | 3.281 | **20.8 %** |
| Colorado, all year | 2.825 | 3.324 | **15.0 %** |
| **Colorado, warm season (May–Sep)** — convective | 3.112 | 3.384 | **8.0 %** |
| **Austin, all year** — convective, flat | 4.583 | 4.919 | **6.8 %** |

This is the paper's central result. **Warm-season Colorado (8.0 %) lands within
1.2 points of flat Austin (6.8 %)**, despite 25.5 % of its cells exceeding 5° slope and
elevations above 4,000 m. Summer mountain precipitation is convective, and convection is
chaotic whether or not there are mountains underneath. The cool season, when
precipitation is orographically forced, reaches **20.8 %** — the only configuration in
this study that meets the 20 % target.

![The forcing ladder](../results/figures/publication/F1_forcing_ladder.png)

*Left: RMSE reduction against bilinear across the four regimes. Right: the decisive
pair — warm-season Colorado and flat Austin score within 1.2 points of each other
despite 25.5 percentage points of difference in steep-cell fraction.*

Skill therefore tracks **how orographically forced the precipitation is**, not how much
relief the domain contains. Terrain is a necessary but not sufficient condition: it
supplies the forcing mechanism only when the synoptic regime engages it.

This also reconciles the two apparent Colorado "failures" in §4.5. Median cell NSE and
the heavy-event POD are annual statistics dominated by the hard convective summer, and
the stage-2 gate (§5.5) behaves identically — positive in the cool season, null in the
warm.

### 4.7 A coarser input: NASA POWER at 0.5° instead of IMERG at 0.1°

The studies above vary the *target*. This one varies the *input*: NASA POWER's
`PRECTOTCORR` (MERRA-2 reanalysis, gauge-bias-corrected, mm day⁻¹) replaces IMERG,
with AORC still the 1 km truth. Three things change at once.

| | IMERG | **NASA POWER** |
|---|---|---|
| nature | satellite retrieval | **reanalysis** |
| grid served | 0.1° × 0.1° | **0.5° lat × 0.625° lon** (native MERRA-2) |
| fine cells per coarse cell | 144 | **3,600** |
| predictors available | 31 | **27** (no half-hour counters) |
| record | 2000-06 → | 1981 → |
| fetch, 21 years | ~46 min of OPeNDAP | **3.3 MB in ~2 s** |

POWER's documentation claims a 0.5° × 0.5° grid; the regional endpoint in fact returns
the native MERRA-2 0.5° × 0.625°, so the input is remapped onto the coarse grid with an
**area-weighted (conservative)** operator — interpolation would not preserve the block
means the downscaling target is defined against.

**Are the input and the target independent?** Not fully, and the detail matters.
MERRA-2's `PRECTOTCORR` is corrected to the CPC Unified gauge analysis, with *full*
correction over 42.5° S–42.5° N land — which includes this entire domain. AORC over
CONUS in our period is primarily Stage IV radar QPE, with CPC only as a secondary source
from 2016 (§2.5). So the two are different products built from different analyses at
different scales, but both are ultimately gauge-informed, since Stage IV is itself
gauge-adjusted at the River Forecast Centres. The comparison with IMERG, an independent
satellite retrieval, is therefore not like-for-like in this respect, and POWER's small
coarse-scale bias (−0.149) should be read with that in mind.

The training domain was widened to `[-101.0, 28.5, -95.0, 35.5]` (168 coarse cells)
because the Austin box alone holds only 42 POWER cells, but **scoring is restricted to
the same Austin box and the same 731 test days** used throughout, so the comparison with
IMERG is exact.

| metric | **POWER ML** | POWER bilinear | **IMERG ML** | IMERG bilinear |
|---|---|---|---|---|
| **RMSE** | **5.679** | 6.303 | **4.583** | 4.919 |
| **reduction** | **9.9 %** | — | **6.8 %** | — |
| bias | +0.288 | −0.149 | −0.046 | −0.082 |
| Pearson *r* | 0.738 | 0.662 | 0.839 | 0.815 |
| NSE | 0.543 | 0.437 | 0.702 | 0.657 |
| KGE | 0.606 | 0.530 | 0.739 | 0.787 |
| POD > 30 mm | 0.382 | 0.288 | 0.548 | 0.601 |
| median cell NSE | 0.552 | 0.443 | 0.711 | 0.661 |

**We predicted a smaller gain and got a larger one.** The reasoning was that
`PRECTOTCORR` is already gauge-bias-corrected, so with less coarse-scale bias to remove
there would be less for the model to recover. The bias column confirms the premise —
POWER's bilinear bias is **−0.149** against IMERG's −0.082, both small — and refutes the
conclusion: the gain rose to 9.9 %.

**The mechanism is scale, not bias.** IMERG at 0.1° already resolves structure down to
~10 km. POWER at 0.5° does not, so a downscaler starting from POWER can recover the
**10–50 km band** — scales that terrain, land cover and local climatology genuinely
constrain. IMERG supplies that band for free, leaving less to win. This refines §6's
claim rather than contradicting it: what is recoverable is *resolved-scale* structure,
and the gain grows with how much of it the coarse input is missing. The sub-grid
component remains out of reach — the 1 km residual stage was rejected an **eighth** time
(6.676 vs 6.683, a 0.10 % gain).

**The absolute product is much worse, which is the practical conclusion.** POWER
downscaled to 1 km scores **5.679**, worse than bilinearly interpolating IMERG with no
model at all (**4.919**). A larger percentage off a much poorer starting point. Where
IMERG is available, downscaling POWER is not worth doing; the result matters for regions
or periods where a 0.1° retrieval is not.

One corroborating detail: **five of POWER's ten most important stage-1 features are
ERA5** (`wind_sin` 4th, `t2m` 6th, `tcwv_max` 8th, `wind_speed` 9th, `cape` 10th),
against roughly two for IMERG. The less the precipitation input resolves, the more the
atmospheric state has to carry — which is what reconstructing resolved-scale structure,
rather than correcting a retrieval, should look like.

**Deep models and a diffusion ensemble on POWER.** All four families were
subsequently trained on this domain and scored on the same Austin box and the same
731 days.

| product | RMSE | bias | KGE | CRPS | POD>30 | spectral ratio <10 km | wet-area ratio |
|---|---|---|---|---|---|---|---|
| IMERG nearest | 6.352 | −0.149 | 0.533 | 2.026 | 0.287 | 3.937 | 1.399 |
| Bilinear | 6.303 | −0.149 | 0.530 | 2.013 | 0.288 | 0.0003 | 1.420 |
| **XGBoost 2-stage** | **5.679** | 0.288 | 0.606 | 2.044 | 0.382 | 0.0006 | 1.871 |
| Swin transformer | 5.886 | 0.230 | **0.636** | 2.022 | 0.402 | 0.061 | 1.626 |
| Diffusion (ens. mean) | 5.947 | 0.310 | 0.635 | **1.534** | 0.394 | 0.203 | 1.597 |
| CNN (U-Net) | 6.078 | 0.584 | 0.617 | 2.135 | **0.509** | 0.284 | 1.701 |
| Diffusion (1 member) | 6.415 | 0.323 | 0.624 | 2.183 | 0.382 | **0.419** | 1.510 |

The ordering reproduces Austin's in every respect that matters: trees take RMSE, the
diffusion ensemble takes CRPS by a wide margin (1.534 against 2.013 for bilinear, 24 %
better than the best deterministic product), and a single diffusion member has the best
texture and nearly the worst RMSE. The CNN again trades RMSE for heavy-event detection,
reaching POD 0.509 against XGBoost's 0.382.

**But the texture recovery is halved.** A single diffusion member retains **0.419** of
observed sub-10 km variance here against **0.840** on Austin — same architecture, same
recipe, same calibration procedure. The difference is the input. Reconstructing
kilometre-scale structure from 50 km squares is a harder inference than from 10 km ones,
and the generative model, which is the only product that attempts it at all, recovers
half as much. The deterministic products are worse still: bilinear retains 0.0003 and
XGBoost 0.0006, both three orders of magnitude short.

Calibration selected the **v** parameterisation (spread score 0.578 against x0's 0.689)
with η = 0 and 40 DDIM steps. Sampled spread reaches 0.602 of observed, so the ensemble
is under-dispersed in the same direction and to a similar degree as on Austin.

![Both inputs compared](../results/figures/publication/F8_input_resolution.png)

*Every product under both coarse inputs, scored on the same Austin box and the same 731
days. Accuracy degrades uniformly with a coarser input while the ordering is preserved;
retained variance (c, logarithmic) collapses for the deterministic products.*

### 4.8 Training the spectrum: texture at no measurable cost in RMSE

Every result above treats the accuracy–realism trade-off as a property of the problem.
It is partly a property of the *objective*. All four `--loss` options in this study are
per-cell losses, and a per-cell loss is minimised by the conditional mean, which is
smooth. The spectral ratio was measured throughout and never optimised.

We added a differentiable spectral penalty to the training objective. It reproduces the
scoring diagnostic in torch — square crop, demean, Hanning window, `|FFT2|²`, radial
average — and penalises the squared gap between predicted and observed **log** power
in the bins below 10 km, averaged over patches that contain rain:

$$\mathcal{L} = \mathcal{L}_{\text{mm-MSE}} \;+\; w \cdot \big\langle (\log P_{\text{pred}} - \log P_{\text{obs}})^2 \big\rangle_{\lambda < 10\,\text{km}}$$

Log power matters: the spectrum spans several orders of magnitude across wavelengths, so
a linear penalty would be decided entirely by the largest scales, which a downscaler
already gets right. The term is **additive**, not a fifth loss *kind* — structure is
orthogonal to the intensity space the loss is taken in. Its torch implementation agrees
with the numpy scorer to a maximum relative difference of 7 × 10⁻¹⁶, so the quantity
trained is the quantity reported.

**The exchange rate is a step, not a slope.** Five weights, Austin, CNN base 24, 6,000
steps, `--select rmse` so the checkpoint criterion cannot confound the comparison:

| *w* | dev RMSE | Δ RMSE | spectral ratio | POD>30 |
|---|---|---|---|---|
| 0 | 4.8612 | — | 0.245 | 0.621 |
| **0.01** | **4.8772** | **+0.33 %** | **1.167** | 0.615 |
| 0.05 | 4.9410 | +1.64 % | 1.206 | 0.627 |
| 0.20 | 4.9484 | +1.79 % | 1.147 | 0.588 |
| 1.00 | 5.1006 | +4.92 % | 1.166 | 0.530 |

`w = 0.01` buys essentially all the available texture; larger weights pay more RMSE for
nothing further, and at `w = 1` the model degrades on every axis.

**Three seeds show the RMSE cost is not detectable.** Paired by seed: +0.0275, −0.0075,
−0.0087, a mean difference of **+0.0038 mm day⁻¹** against a seed standard deviation of
0.040 (*t* ≈ 0.32 on 2 df). Two of three seeds were *better* with the penalty; the
+0.33 % in the sweep above was seed 0 being unlucky. Mean spectral ratio rose 0.365 →
1.052, every seed gaining several times the spread. The penalty also **stabilises**
texture: the control ranges 0.245–0.476 across seeds, the penalised runs 0.988–1.112.

This test is lightly powered — with *n* = 3 and σ ≈ 0.04 it can only detect differences
above roughly 0.05 mm day⁻¹, so a real cost of that size could hide. The 30,000-step run
suggests one of about that magnitude: the control reaches 4.8291 at step 21,000 while the
penalised run peaks at 4.8689 at step **7,000** and then overfits. The honest statement
is *no detectable cost at matched budget*, with the penalty reaching its optimum three
times sooner.

**On the test set, three seeds, against the matched control:**

| configuration | RMSE | spectral ratio | POD>30 | CSI>30 |
|---|---|---|---|---|
| CNN, matched control | 4.655 ± 0.007 | 0.378 ± 0.083 | 0.531 ± 0.011 | 0.442 ± 0.007 |
| **CNN + spectral penalty** | 4.675 ± 0.012 | **1.056 ± 0.080** | 0.522 ± 0.005 | 0.438 ± 0.003 |

Paired by seed: ΔRMSE **+0.0198 ± 0.0127** (*t* = 2.70), ΔPOD **−0.0096 ± 0.0062**
(*t* = −2.68), ΔCSI −0.0043 (*t* = −1.68).

**The dev-year result does not survive intact.** On dev the penalty was free; on test it
costs about 0.02 mm day⁻¹ of RMSE and, more importantly, **a small but consistent amount
of heavy-rain detection** — POD falls in all three seeds. The texture gain is real and
large (0.378 → 1.056, every seed gaining several times the spread), and the penalty still
stabilises it: the control ranges 0.287–0.450 across seeds while the penalised runs range
1.007–1.148. But the honest statement is that realistic texture costs a little detection,
not that it is free. §4.9 shows that cost can be repaid.

A caution for anyone reproducing this. Two runs with **identical seed and identical
settings** gave 4.8772 and 4.8887 on dev with the penalty active, while the `w = 0`
control reproduced to the digit. The FFT's reductions are not deterministic on GPU, so
the spectral term adds run-to-run variation beyond the seed, and single-run comparisons
of it are unsafe in a way the per-cell losses are not.

**Is the added variance in the right places, or is it well-sized noise?** The spectral
ratio is an amplitude diagnostic and cannot distinguish the two. FSS can. Against the
matched control, at every neighbourhood width:

| threshold | 1 cell | 5 cell | 15 cell | 41 cell |
|---|---|---|---|---|
| 1 mm | **+0.0046** | +0.0061 | +0.0066 | +0.0065 |
| 10 mm | −0.0011 | +0.0005 | +0.0002 | −0.0004 |
| 30 mm | −0.0026 | −0.0007 | −0.0005 | −0.0005 |

Light-rain placement improves at **every** scale including a single cell, which random
speckle of the correct magnitude would have degraded. Wet-area ratio also moves 1.252 →
1.193, so the model drizzles across fewer cells that stayed dry. But the heavy-rain row
is uniformly, if marginally, worse, and POD fell in all three seeds. **The added
structure is light-rain structure.** This is a realism result, not an extremes result.

![Placement skill](../results/figures/publication/F7_fss_placement.png)

*Change in Fractions Skill Score against the matched control. Improvement at a single
cell for the 1 mm threshold is the test that matters: noise of the correct magnitude
would lower it.*

One qualification. The test checkpoint **overshoots**, at 1.260 against a target of 1.0;
measured as distance from correct, a single diffusion member (0.840) remains the
better-calibrated field. The dev runs land at 1.052, so the penalty can hit the target,
and the likely cause of the overshoot is that it trains on 96-cell patches while the
diagnostic is measured on 420 × 360 fields.

**The same pair on the coarser input.** POWER is the harder case — the deterministic
products there retain 0.0003 to 0.284 of observed sub-10 km variance and even a diffusion
member reaches only 0.419. A matched control/spectral pair, identical but for the weight,
trained on that domain and scored on the same Austin box and 731 days:

| | RMSE | spectral ratio | POD>30 | CSI>30 | KGE | FSS 10 mm@15 km |
|---|---|---|---|---|---|---|
| CNN, matched control | **5.782** | 0.167 | 0.391 | 0.306 | 0.633 | 0.670 |
| **CNN + spectral penalty** | 5.821 | **1.281** | **0.414** | **0.314** | **0.651** | **0.675** |

**7.7× the texture for +0.68 % RMSE, and every other metric improves.** Unlike Austin,
heavy-event detection *rises* here (POD 0.391 → 0.414, CSI 0.306 → 0.314), as do KGE and
FSS. The result is stronger than on Austin in relative terms, because nothing else on
this domain comes close to correct texture: the penalised CNN has the second-best RMSE of
any POWER product, behind only XGBoost's 5.679, and three times the fine-scale variance
of the next-best product of any kind.

The +0.039 mm day⁻¹ RMSE difference is the same order as the Austin seed spread (0.040)
and we did not repeat seeds on this domain, so we report it as a point estimate rather
than a bounded cost.

**A caveat on how the checkpoint was obtained.** Selection here is plain `--select rmse`,
and the minimum landed at step 4,000, where dev texture was 1.31. Later checkpoints are
smoother: by step 40,000 dev texture has decayed to 0.54 while dev RMSE is flat. On
Austin the penalty held above 1.0 throughout training; on POWER the optimiser gives
texture back as training proceeds, and the outcome depends on the RMSE optimum falling
early. A selection criterion with a spectral floor — the analogue of `rmse_pod` — would
make this robust rather than fortunate, and we have not built one.

Note also that this control is **not** the CNN row of §4.7. That model used `rmse_pod`
selection and scores 6.078 with texture 0.284; this one uses plain RMSE selection and
scores 5.782 with texture 0.167, a smoother checkpoint at better RMSE. The pair above is
internally consistent and should not be read against the §4.7 table.

F3 in §4.2 places this product against all the others: it is the only point that sits
both at the left edge, with the accurate products, and on the dashed line.

![The spectral exchange rate](../results/figures/publication/F5_spectral_exchange.png)

*(a) Best checkpoint per weight — the knee is at w = 0.01. (b) Texture during training
for the matched pair at 30,000 steps.*


### 4.9 Recovering both: structure *and* detection

§4.8 buys texture at a small cost in heavy-rain detection. Two further interventions
repay it — one trained, one post-hoc — and together they turn a single compromised
product into a menu.

**Combining the two things that worked.** Heavy-day upweighting (§5.4) buys +0.030 POD
at no RMSE cost; the spectral penalty buys texture at a small POD cost. They act on
different things and compose. A third arm conditions the spectral penalty on heavy
crops instead, aiming the texture where detection lives. Test set, three seeds each,
all against the same matched control:

| configuration | RMSE | spectral ratio | POD | FAR | CSI | freq. bias | FSS30@5 |
|---|---|---|---|---|---|---|---|
| control (MSE) | 4.655 ± 0.007 | 0.378 ± 0.083 | 0.531 ± 0.011 | 0.276 | 0.442 ± 0.007 | 0.733 | 0.640 |
| + spectral | 4.675 ± 0.012 | 1.056 ± 0.080 | 0.522 ± 0.005 | 0.269 | 0.438 ± 0.003 | 0.714 | 0.637 |
| **+ spectral + heavy ×3** | **4.654 ± 0.007** | **1.027 ± 0.100** | **0.594 ± 0.006** | 0.323 | **0.463 ± 0.001** | 0.878 | **0.661 ± 0.001** |
| + heavy-*conditioned* spectral | 4.638 ± 0.019 | 1.893 ± 0.153 | 0.539 ± 0.011 | 0.280 | 0.446 ± 0.006 | 0.749 | 0.646 |

Paired by seed against the control. The combined configuration was subsequently
extended to **five seeds**; the other two arms remain at three:

| | ΔRMSE | ΔPOD | ΔCSI |
|---|---|---|---|
| + spectral *(n=3)* | +0.0198 ± 0.0127 (*t* = 2.70) | −0.0096 ± 0.0062 (*t* = −2.68) | −0.0043 (*t* = −1.68) |
| **+ spectral + heavy ×3 *(n=5)*** | **−0.0035 ± 0.0200 (*t* = −0.39)** | **+0.0646 ± 0.0074 (*t* = 19.39)** | **+0.0214 ± 0.0060 (*t* = 8.03)** |
| + heavy-conditioned *(n=3)* | −0.0168 ± 0.0126 (*t* = −2.31) | +0.0083 ± 0.0137 (*t* = 1.05) | +0.0042 (*t* = 1.40) |

**The combination is the result of this section.** At no RMSE cost whatever
(*t* = −0.39) it delivers correct texture — 1.007 ± 0.105 over five seeds, the closest
to 1.0 of any configuration in the study, trained or post-hoc — together with
**+0.065 POD at *t* = 19.4** and +0.021 CSI at *t* = 8.0. POD rises in **all five
seeds**; ΔRMSE changes sign across them, which is what an absent effect looks like.
A day-block bootstrap over the 731 test days, which respects the clustering of heavy
cells into a few dozen convective days, puts ΔPOD at **[+0.031, +0.075]** and ΔCSI at
**[+0.0002, +0.036]**; both exclude zero.

**How large an effect has to be before we believe it.** Five control runs differing
only in seed span RMSE 4.6380–4.6814 (sd **0.0163**) and POD 0.5189–0.5421
(sd **0.0097**). Two *controls* compared against each other with the day-block
bootstrap give ΔRMSE −0.043 and ΔPOD +0.016, **both with intervals excluding zero** —
entirely from the seed. The bootstrap resamples days, not initialisations, so on its
own it will certify seed noise. Every difference reported in this section is therefore
paired within seed, and read against that spread: the combined configuration's ΔPOD is
6.6× the control POD sd, while its ΔRMSE is a fifth of the control RMSE sd.

Detection under this configuration is also unusually stable: POD varies by 0.006 across
seeds and CSI by 0.001, against a control whose *texture* alone varies by 0.083.

**Conditioning the penalty on heavy rain fails at what it was designed for.** It
over-roughens badly (1.893, further from correct than any other configuration) and its
detection gain is not significant. Restricting the spectral term to heavy-bearing crops
starves it of the light-rain samples that anchor the spectrum. It does buy RMSE
(−0.0168, *t* = −2.31), so it is a trade, not a failure on every axis — but not the
trade it was built to make.

**Two cheaper routes to the same place, both null.** If the gain were really about
tolerating displacement, a neighbourhood loss should reach it without a spectral term.
Multiscale MSE on average-pooled fields at 1/3/9/27 cells gives the best RMSE in the
set (4.637) and **moves nothing else**: ΔPOD +0.0025, texture 0.287 → 0.356. Added on
top of the combined configuration it reproduces that configuration's POD to four
decimal places (+0.0001). Pooling tolerates displacement; it does not ask for structure.

The second route was selection rather than loss: keep the best-RMSE checkpoint *among
those already above a texture floor of 0.8* (`--select rmse_spec --spec-floor 0.8`).
At one seed this was the best RMSE/POD pair in the study (4.642, POD 0.610, both better
than the combined configuration at that seed). **Across five seeds it is a null**:
ΔPOD +0.0025 (*t* = 0.6), ΔRMSE −0.0020 (*t* = −0.4). At two of the five the floor
never binds — the best-RMSE checkpoint already cleared it, so the selected weights, and
hence every score, are identical. The +0.018 at seed 0 was inside the ±0.0097 control
spread, and is the clearest illustration in this study of why the preceding paragraph's
discipline is not optional.

**Post-hoc: quantile mapping.** The *intensity distribution* a squared-error fit
compresses needs no retraining. A map fitted on the dev year and applied to test is
monotone, so every cell keeps its rank and only the values move.

| | RMSE | spectral ratio | POD | FAR | CSI | freq. bias | KGE | wet-area |
|---|---|---|---|---|---|---|---|---|
| XGBoost 2-stage | **4.583** | 0.021 | 0.548 | **0.262** | 0.458 | 0.742 | 0.739 | 1.320 |
| **XGBoost, quantile-mapped** | 4.643 | 0.048 | 0.636 | 0.322 | **0.488** | **0.938** | **0.809** | **0.953** |

CSI 0.488 and KGE 0.809 are the best figures in the study. The cost is +0.060 mm day⁻¹,
which unlike §4.8's +0.004 is a real trade — eight times the four-seed XGBoost spread.

**Its value is proportional to the frequency bias it corrects**, which makes it a
shrinkage correction rather than a general improvement. XGBoost at 0.742 gains most;
Swin at 0.981 gains nothing on CSI (0.469 → 0.469) while losing 0.083 RMSE.

**Post-hoc: probability-matched mean.** Pattern from the ensemble mean, intensity
distribution from the pooled members (Ebert, 2001). Spectral ratio **1.055** — the
best-calibrated texture anywhere in this study — for zero training, at the cost of RMSE
(5.062) and detection (CSI 0.404).

**The two corrections must not be stacked.** Quantile mapping on top of a
spectral-penalty model pushes texture to 1.74, and on top of the combined configuration
to 1.90, without improving CSI. They are orthogonal in what they touch — one the values,
one their spatial arrangement — but a field that already carries the right fine-scale
variance does not want its tail inflated as well.

**Does any of this survive terrain?** The combined configuration was repeated on the
Colorado Front Range, three seeds, paired within seed:

| | RMSE | spectral ratio | KGE | POD@14.5 mm | freq. bias |
|---|---|---|---|---|---|
| control (MSE) | 2.978 ± 0.003 | 0.351 ± 0.019 | 0.329 ± 0.011 | 0.127 | 0.146 |
| **+ spectral + heavy ×3** | **2.938 ± 0.020** | **1.160 ± 0.189** | **0.374 ± 0.017** | **0.159** | 0.191 |

**Texture replicates and is no longer a flat-terrain result**: 0.351 → 1.160, a factor
of 3.3 against Austin's 3.0. RMSE improves in all three seeds (−0.040; *t* = −3.1,
short of the *t* = 4.30 that three seeds demand, but one-signed), where in Austin it was
null — so the objective is not merely free in terrain, it may be favourable. KGE +0.046.

**Detection needed a threshold correction before it could be read at all.** At 30 mm the
Colorado control scores POD 0.013 at a frequency bias of **0.039** — it forecasts the
event 4 % as often as it occurs, and POD, CSI and FSS all collapse into a corner that
describes the threshold rather than the model. 30 mm is **6.7× rarer** over the Front
Range than over Austin. Rescoring at the depth that reproduces Austin's exceedance rate
(**14.50 mm**, Colorado's 98th percentile) makes the comparison meaningful, and the gain
holds: ΔPOD **+0.0317** (*t* = 4.35), ΔCSI **+0.0290** (*t* = 4.39), about half the
Austin effect in the same direction.

**But the correction does not rescue the regime, and that is itself a result.** Even at
matched exceedance the Colorado frequency bias is 0.15 against Austin's 0.73. Both
configurations under-forecast heavy rain in terrain by roughly fivefold, so the
cross-domain claim is that the objective *helps* where detection is poor, not that it
makes detection good there. We report detection in this domain at the matched threshold
throughout, and the score files record the threshold they were produced with.

**A menu, not a winner.**

| if the decision needs… | use | evidence |
|---|---|---|
| lowest error | XGBoost | RMSE 4.583 |
| heavy-rain detection | XGBoost + quantile map | CSI 0.488, KGE 0.809 |
| structure *and* detection | **spectral + heavy ×3** | texture 1.007, POD 0.597, no RMSE cost, 5 seeds + 3 in terrain |
| a realistic field | PM mean of the diffusion ensemble | texture 1.055, no training |
| a calibrated distribution | diffusion ensemble | CRPS 1.050, 28 % better than any deterministic product |

This is §4.2's argument made concrete. The conditional mean, a draw from the conditional
distribution, and an upper quantile are three different estimators serving three
different decisions, and the right response is to ship the one the decision needs rather
than to seek a single raster that is all three.


---

## 5. Ablations and negative results

These are the paper's main empirical content. Unless stated, ablations are scored on the
2018 dev year of the Austin domain, where the bilinear reference is RMSE 5.1525 and
POD>30 0.6329.

### 5.1 Capacity and architecture do not matter

| model | parameters | best dev RMSE |
|---|---|---|
| CNN base 24 | 1.65 M | 5.1365 |
| CNN base 64 | 11.7 M | 5.1382 |
| Swin transformer | 7.9 M | 5.1389 |

A 7× parameter increase and a change of inductive bias from convolution to windowed
attention move dev RMSE by **0.0024 mm day⁻¹ (0.05 %)**. On the test set the four
families — including gradient-boosted trees, a fundamentally different hypothesis class —
span 4.583 to 4.807, a 4.9 % range. Excluding the generative model, which optimises a
different objective, the three deterministic families span just **1.7 %**.

### 5.2 What did move the numbers: information

| intervention | effect on test RMSE |
|---|---|
| ERA5 environmental predictors | −2.2 % |
| record extended 4 → 18.5 years | −2.0 % |
| sub-daily structure features (27 → 31 channels) | −1.5 % |

Per-family effect of the sub-daily features:

| model | 27 features | 31 features |
|---|---|---|
| XGBoost | 4.655 | **4.583** |
| CNN | 4.694 | 4.663 |
| Swin | 4.729 | **4.653** |
| Diffusion (ens. mean) | 4.840 | 4.807 |

Margin over bilinear moved 5.4 % → 6.8 %. `imerg_cond_intensity` ranks 4th of 31 and
`imerg_wet_frac` 5th in Austin, above every ERA5 field. In Colorado `imerg_wet_frac`
ranks 3rd and `imerg_cond_intensity` 8th, so the features transfer across regimes.

Per-family gains of 0.7–1.6 % come from single runs; §5.8 bounds the noise at 0.15 %
(trees) and ~0.8 % (both deep models). The XGBoost gain (1.5 %) clears its noise floor by
an order of magnitude and the Swin gain (1.6 %) by about 2×; the CNN (0.7 %) and
diffusion (0.7 %) gains sit at or below theirs and should not be read as established.

### 5.3 Loss space: optimise in the space you score in

| loss | best dev RMSE |
|---|---|
| log-space MSE | 5.5641 |
| hybrid (0.5 log + 0.5 mm) | 5.2858 |
| **mm-space MSE** | **5.1752** |

These are 1,000-step runs, so all three sit above the bilinear reference; the comparison
is between losses, not against interpolation.

Log-space MSE is the conventional choice for skewed precipitation, and it produced
training curves that fell monotonically while dev RMSE in mm *rose*. The model was
correctly minimising an objective that was not the reported metric. This is worth
stating plainly because the failure is silent: nothing in the training log indicates it.

### 5.4 Extreme-event remedies: all null

Six configurations, identical except for the loss or sampling treatment:

| configuration | best dev RMSE | best dev POD>30 |
|---|---|---|
| base (mm MSE) | 5.1365 | 0.6119 |
| quantile τ = 0.90 | 5.1369 | 0.6164 |
| quantile τ = 0.95 | 5.1376 | 0.6175 |
| importance weighting w(p) | 5.1378 | 0.6150 |
| heavy-day upweight ×3 | 5.1379 | **0.6417** |
| uniform crop sampling | **5.1336** | 0.6180 |

RMSE spans **0.0043 mm day⁻¹ across all six** — indistinguishable, and below the CNN
seed noise of §5.8. Heavy-day upweighting buys +0.030 POD at no RMSE cost and is the
only treatment with a defensible effect.

**The 0.80 target is not reachable from this input and should not be reported as a
miss.** Bilinear interpolation of IMERG already achieves POD 0.601; the best
configuration anywhere in this study reaches 0.636 (§4.9). A daily 10 km average does
not determine which square kilometre received the storm core (§5.5), so pointwise
detection of 30 mm events is bounded well below 0.80 by the information in the input,
not by the treatment of the loss. A detection target for this task should be set on CSI
or on a neighbourhood score, both of which the learned products do improve.

### 5.5 The 1 km residual stage: rejected in every configuration, in both domains

The residual stage is gated on a training-tail hold-out and adopted only if it beats
bilinear by more than 0.5 %. It has now been rejected **seven times**:

| configuration | hold-out residual | hold-out bilinear | gain | decision |
|---|---|---|---|---|
| Austin, 4 years | — | — | < gate | rejected |
| Austin, 18.5 years, no ERA5 | — | — | < gate | rejected |
| Austin, 18.5 years, with ERA5 | 5.167 | 5.176 | 0.17 % | rejected |
| Wide domain (2.4× area) | 4.559 | 4.559 | 0.00 % | rejected |
| **Colorado, all year** | 2.583 | 2.588 | **0.19 %** | rejected |
| **Colorado, cool season** | **2.394** | **2.399** | **0.21 %** | rejected |
| **Colorado, warm season** | 2.802 | 2.801 | **−0.04 %** | rejected |

The seasonal split is the informative part. In the **cool season** the residual model is
consistently better than bilinear on both the hold-out (0.21 %) and the validation period
(2.583 vs 2.594, 0.42 %), and its importances are dominated by terrain:

```
dem, dem_anom, pred_bl, imerg_bl, imerg_pct_bl,
imerg_roll7_bl, lc_frac_water, doy_cos, dem_x_forest, doy_sin
```

`dem` first and `dem_anom` second. In the **warm season** the residual model is
*marginally worse than bilinear* on the hold-out, and terrain drops out of the top two.

So sub-grid placement is **detectable** where orography forces it — the sign is
consistent, the mechanism is identifiable in the importances, and it strengthens in
exactly the season theory predicts. But the magnitude is 0.2–0.4 %, an order of
magnitude below what would justify shipping a second model stage. A 10 km average,
terrain and land cover still do not determine which square kilometre received the storm
core, even at 25.5 % steep terrain in the season most favourable to the hypothesis.

### 5.6 Model stacking: dev gain did not transfer

Non-negative least squares over all five products, fitted on the 2018 dev year with an
explicit no-leakage check:

```
weights   XGBoost 0.360   CNN 0.621   Swin 0.0   Diffusion 0.0   Bilinear 0.0
dev RMSE  blend 4.937   best member (CNN) 4.973   equal-weight 4.972
```

The blend improved the dev year by 0.036 mm day⁻¹, but on the test set scored **4.601
against XGBoost's 4.583** — worse than its best member. The result reproduced with a
completely different weight vector from an earlier feature generation, and the gauge
validation agrees independently (stacked 6.848 vs XGBoost 6.754). With ~365 independent
dev days, NNLS weights do not generalise.

### 5.7 Terrain transfer: learned, and still useless on a flat target

Before the Colorado domain was built, we tested whether *training* on terrain would help
a flat *target*. The training domain was widened westward from `[-99, 28.5, -96, 32]` to
`[-104, 28.5, -96, 32]` — 2,800 coarse cells (2,504 valid) against 1,050, and 16,083,192
training rows against ~6.7 M — while scoring **only on the original Austin box**.

| longitude band | mean elevation | max elevation | fraction of cells > 5° slope |
|---|---|---|---|
| W097 (Austin) | 100 m | 196 m | **0.000** |
| W102 | 695 m | 909 m | 0.187 |
| W104 | 1,299 m | 2,277 m | 0.210 |

| metric | wide-trained | Austin-trained | Δ |
|---|---|---|---|
| **RMSE** | **4.609** | **4.583** | **+0.026 (worse)** |
| MAE | 1.466 | 1.467 | −0.001 |
| NSE | 0.699 | 0.702 | −0.003 |
| KGE | 0.737 | 0.739 | −0.002 |
| reduction vs bilinear | 6.3 % | 6.8 % | worse |

The bilinear control scores 4.917 against 4.919, confirming identical scoring data.

**The model did learn orography and it still did not help.** Residual-stage importances
on the wide domain place `dem` 3rd, `slope` 9th and `dem_x_slope` 10th. But Austin
contains no cells above 5° slope, so an orographic relationship learned in west Texas
has no Austin feature values on which to act, while ~1,450 extra western cells dilute a
second, arid regime into the same tree split budget.

This experiment is what motivated §4.5: the relief has to be **inside** the evaluation
box, not merely inside the training set. Put there, the same relationship produces
15.0 % instead of −0.6 %.

**The reverse transfer closes the design and identifies what is actually learned.**
Applying the *Colorado-trained* stage-1 model to *Austin* features completes the four
cells:

| trained on | scored on | RMSE | vs bilinear | says |
|---|---|---|---|---|
| Austin | Austin | 4.583 | **+6.8 %** | baseline |
| Texas + west | Austin | 4.609 | −0.6 % | terrain in *training* does not help a flat target |
| Colorado | Colorado | 2.825 | **+15.0 %** | terrain in *scoring* more than doubles skill |
| **Colorado** | **Austin** | **5.593** | **−13.7 %** | **what is learned is domain-specific** |

The cross-domain model is **13.7 % worse than bilinear interpolation** — it actively
destroys skill rather than merely failing to add any. The mechanism is in the bias:

| | Colorado model on Austin | Austin's own model | bilinear |
|---|---|---|---|
| RMSE | 5.593 | 4.583 | 4.919 |
| **bias** | **+0.353** | −0.046 | −0.082 |
| MAE | 2.209 | 1.467 | 1.506 |
| KGE | 0.522 | 0.739 | 0.787 |
| POD > 30 mm | 0.283 | 0.548 | 0.601 |

Had the terrain features simply been out of distribution — Austin spans 100–668 m
against Colorado's training range of 1,221–4,245 m, so no tree split can extrapolate —
the model would have fallen back on its IMERG-equivalent predictors and landed near the
6.8 % Austin baseline. Instead bias swings from −0.082 to **+0.353**: it systematically
over-predicts rain.

That is precisely what it was trained to do. In Colorado the retrieval badly
under-reports (bilinear bias −0.687, raw correlation 0.573), so the stage-1 model's
dominant learned behaviour is *add precipitation*. Austin's retrieval is nearly unbiased.
Applying a correction calibrated for a severe dry bias to an input that does not have one
produces a wet bias of similar magnitude.

This is the strongest available confirmation of §6's claim that these models recover
**coarse-scale bias correction**, not transferable sub-grid physics. A model that had
learned general orographic relationships would degrade gracefully on flat terrain; one
that learned "this retrieval is 0.69 mm day⁻¹ too dry here" fails in exactly this way.

### 5.8 Seed repeats: how large a difference is interpretable

Every model comparison above came from a single training run. We repeated the three
families whose rankings the results appeared to decide: ten runs in total.

| family | seeds | test RMSE | mean | spread |
|---|---|---|---|---|
| **XGBoost** | 1, 2, 3, 42 (published) | 4.578 / 4.583 / 4.585 / 4.583 | 4.5823 | **0.007 (0.15 %)** |
| **CNN (1.65 M)** | 1, 2, 0 (published) | 4.701 / 4.669 / 4.663 | 4.6776 | **0.037 (0.80 %)** |
| **Swin (7.9 M)** | 1, 2, 0 (published) | 4.614 / 4.646 / 4.653 | **4.6377** | **0.039 (0.84 %)** |

Deep-model seeds match the published run's **step budget** (12,500 for Swin, 40,000 for
the CNN), not its wall-clock: the original Swin run was time-limited at 12,494 steps, so
matching minutes across different GPUs would confound hardware with seed.

![Seed spread](../results/figures/publication/F6_seed_repeats.png)

*(a) Seed-to-seed range for each family; the published CNN–Swin pairing is four times
smaller than the difference of means. (b) The spectral comparison of §4.8, paired within
seed — the group means are indistinguishable and two of three seeds favour the penalty.*

Three conclusions.

**The headline 4.583 is reproducible.** Four XGBoost seeds span 0.15 %, against a 6.8 %
margin over bilinear — a signal-to-noise ratio near 45:1. The tree result is not a lucky
draw.

**The CNN–Swin ordering is real, and the single run understated it fourfold.** Swin is
better in **9 of 9 pairwise seed comparisons** and the two ranges are disjoint — Swin's
worst (4.6530) beats the CNN's best (4.6630). The difference of means is **0.0399 mm
day⁻¹ (0.85 %)**, against the published single-run gap of **0.0100**: that run paired the
CNN's best seed with Swin's worst. An exact rank test on complete separation with three
runs each gives one-sided *p* = 1/20 = 0.05, the smallest value this design can produce —
consistent, but not strongly powered.

This cuts against the comfortable reading of single-run noise. Such comparisons are
unreliable **in both directions**: they can invent a difference that is not there, and —
as here — hide one that is. Our own first reading of this table concluded the ordering
was not real; three seeds per family reversed it.

**Trees are five times steadier than either deep model.** XGBoost spans 0.15 % where the
CNN and Swin span 0.80 % and 0.84 % on identical data and splits. That is optimisation
stochasticity on ~1,100 effective wet days, and it is why deep-model differences need
seed support that tree differences do not.

---

## 6. Discussion

**Skill tracks orographic forcing, not terrain and not model capacity.** The four-way
ladder — Colorado cool 20.8 %, Colorado all-year 15.0 %, Colorado warm 8.0 %, Austin
6.8 % — is monotone in how deterministically the terrain decides where precipitation
lands, and is *not* monotone in how much terrain is present (the 8.0 % and 6.8 % figures
come from domains differing by 25.5 percentage points of steep-cell fraction). Capacity
varied 7× across architectures for a 0.05 % effect.

**What is recoverable is coarse-scale, not sub-grid, and it is domain-specific.** The
cross-domain test (§5.7) makes this concrete: a Colorado-trained model scores −13.7 %
against bilinear on Austin, swinging bias from −0.08 to +0.35, because what it learned
was that *its own* retrieval runs dry — not how orography shapes rainfall. Even in the most favourable
configuration tested — 25.5 % steep cells, cool season, terrain ranked first and second
in the residual model's importances — the 1 km residual stage gains 0.2–0.4 % and is
rejected. The gains that *are* real come from bias correction and neighbourhood
structure at 10 km. This is why Colorado's domain-wide RMSE improves 15 % while its
median per-cell NSE (0.490) is *worse* than flat Austin's (0.711): the model removes a
large systematic error without making individual square kilometres predictable.

**A coarser input yields a larger relative gain and a worse product.** NASA POWER at
0.5° gives 9.9 % against bilinear where IMERG at 0.1° gives 6.8 %, yet POWER downscaled
(5.679) is worse than IMERG merely interpolated (4.919) — see §4.7. The extra gain comes
from reconstructing the 10–50 km band that IMERG already resolves, not from recovering
anything below 1 km. Percentage improvement over interpolation is therefore not
comparable across inputs of different resolution, and should not be reported as if it
were.

**The literature's 15–25 % figures are regime statements, not method statements.** Our
Colorado cool season reproduces them (20.8 %) using gradient-boosted trees and 31
features. Our Austin domain, with the same code, reaches 6.8 %. Reporting a downscaling
method's skill without characterising the forcing regime of its evaluation domain
conveys little.

**Accuracy and realism must be reported jointly — and part of the trade-off is the
objective, not the problem.** A product can achieve the best RMSE while retaining 2 % of
observed fine-scale variance. Reported alone, RMSE would select exactly the product least
suitable for hydrological forcing, where spatial gradients drive runoff. We suggest
spectral ratio below the native resolution as a required diagnostic, precisely because it
cannot be improved by smoothing.

The stronger claim we had drawn from this — that the two cannot be jointly maximised —
does not survive §4.8. Every loss in the study was a per-cell loss, and a per-cell loss
is minimised by the conditional mean; the diagnostic was measured and never optimised.
Once the spectrum enters the objective, a deterministic CNN reaches 1.26 retained
variance at 4.635 RMSE on the 10 km input, where previously that order of texture
required a diffusion model at 5.483 — worse than bilinear. On the 50 km input the same
change takes retained variance from 0.167 to 1.281 for 0.68 % of RMSE, and *improves*
heavy-event detection, KGE and FSS at the same time. The remaining trade-off is real but
far narrower than the deterministic/generative framing implies. Which metrics pay for the
texture appears to depend on the regime rather than being fixed: on Austin heavy-event
detection slips marginally, on POWER it gains. A generative product is still the right answer where a
calibrated distribution is needed, which CRPS (1.050 on Austin, 1.534 on POWER, both
~25 % better than any deterministic product) continues to show.

**Effective sample size deserves explicit statement.** Reporting "970 million training
samples" for 6,423 days × 151,200 cells is misleading by three orders of magnitude.
Models overfit within a few thousand optimisation steps irrespective of capacity, which
is the behaviour expected from ~6,400 independent units, of which ~1,100 are
meaningfully wet — and it is why the CNN's seed spread (0.8 %) is five times the tree's.

---

## 7. Limitations

1. **Two domains, not a survey.** The regime hypothesis rests on two boxes plus a
   seasonal split within one of them. The seasonal decomposition is the stronger
   evidence, because it varies forcing while holding geography exactly fixed, but
   replication across more domains would strengthen the claim.

2. **Deep models were run on the flat domain only.** The Colorado results are from the
   tree pipeline. Whether the CNN, Swin and diffusion models gain proportionally in an
   orographic regime is untested, and the accuracy–realism trade-off of §4.2 has not
   been re-measured there.

3. **Seed repeats are lightly powered.** XGBoost (4 seeds), CNN and Swin (3 each) are
   bounded; diffusion is not. The CNN–Swin separation rests on three runs per family,
   where complete separation can only reach *p* = 0.05; five seeds each would settle it.
   Diffusion differences remain uninterpretable.

4. **Snow adds a confound in Colorado.** Cool-season precipitation is largely frozen and
   IMERG is weak over snow — visible in `imerg_prob_liquid` rising from 31st to 7th in
   importance. The cool season is therefore both the most orographically forced *and*
   the most retrieval-degraded period; we interpret its 20.8 % as orographic skill, but
   the two effects are not fully separated.

5. **Daily temporal resolution.** Sub-daily downscaling is where the structural
   information is richest, and we access it only through daily-aggregated counters.

6. **Partly dependent validation.** AORC assimilates gauge observations, so the GHCN-D
   check in §4.3 is a consistency test, not an independent one, and it was run on the
   Austin domain only.

7. **The POWER comparison varies three things at once.** Resolution (0.5° vs 0.1°),
   data type (reanalysis vs retrieval) and predictor count (27 vs 31, no sub-daily
   structure) all differ, so §4.7 attributes the larger gain to resolution by mechanism
   and by the feature-importance shift, not by a controlled isolation of it.

8. **The reference is not a single product.** AORC changes precipitation methodology
   twice inside our record and stops bias-adjusting entirely in 2016 (§2.5), so train
   and test sit on opposite sides of that break. We measured the reference's own
   statistics by year and found no step above one interannual standard deviation, but
   the test is on one domain, on the 40 wettest days of each year, and would not detect
   a change confined to a season, a region or to structure that these summary statistics
   miss. Separately, scoring after 2016 means scoring against unadjusted radar QPE.

9. **The spectral penalty overshoots, and its checkpoint is not robustly selected.**
   Both domains reach ~1.3 on the test set against a target of 1.0, trading
   over-smoothing for mild over-roughening; by distance from correct a single diffusion
   member remains better calibrated. The penalty trains on 96- or 180-cell patches while
   the diagnostic is evaluated on full fields, which is the likely cause and is not
   fixed. On POWER the texture decays through training and the result depends on the
   RMSE optimum falling early — there is no spectral analogue of the `rmse_pod` floor.
   Seeds were repeated on Austin only (*n* = 3, detectable cost ≳0.05 mm day⁻¹); the
   POWER +0.68 % is a single point estimate.

10. **Targets met in one regime only.** The 20 % RMSE reduction target is met in the
   Colorado cool season (20.8 %) and missed everywhere else. The POD ≥ 0.80 target for
   events above 30 mm day⁻¹ is missed in both domains, badly so in Colorado (0.051).

---

## 8. Conclusions

Downscaling IMERG from 10 km to 1 km yields **6.8 %** RMSE reduction against bilinear
interpolation over flat, convectively-dominated central Texas, and **15.0 %** over the
Colorado Front Range — two domains constructed to be identical in size, period, splits
and predictors, differing only in terrain. Decomposing the second by season resolves the
mechanism: **20.8 %** in the orographically forced cool half-year, **8.0 %** in the
convective warm half-year, the latter statistically indistinguishable in character from
the flat domain's all-year result. Downscaling skill is a function of how deterministic
the forcing is, not of how much relief the domain contains.

Within any single regime, method does not matter. Four model families spanning
gradient-boosted trees to a 12.4 M-parameter diffusion model converge to within 4.9 % of
each other; seed repeats place the noise floor at 0.15 % for trees and ~0.8 % for both
deep models, and show the published CNN–Swin gap to have understated the true difference
fourfold. Three
informational additions — ERA5 predictors, a 4.6× longer record, and four sub-daily
structure features obtained free from the daily granule — account for the entire margin.

Seven attempts to recover the sub-grid component failed, including in the orographic
season where terrain ranks first and second in the residual model's importances and the
gain is consistently positive at 0.2–0.4 %. We conclude that satellite precipitation
downscaling at this scale recovers coarse-scale bias and neighbourhood structure, not
sub-grid placement; and that the apparent disagreement between published studies
reporting 15–25 % and our 6.8 % is explained by evaluation regime rather than method.

Separately, the best-RMSE product retains 2 % of observed sub-10 km variance while the
most realistic has the worst RMSE, so studies reporting RMSE alone will systematically
select over-smoothed products. That trade-off is, however, partly an artefact of
optimising a per-cell loss while only *measuring* the spectrum. Adding a differentiable
spectral penalty raises retained variance to 1.26 at no RMSE cost detectable against a
three-seed noise floor, and improves light-rain placement at every neighbourhood width.
Realistic texture therefore does not require a generative model; a calibrated predictive
distribution still does.

The one methodological contribution that transfers directly is the sub-daily structure
features: four predictors recovering convective-versus-stratiform character from
counters already present in the daily IMERG granule, at essentially zero additional
download cost, which improved every model family tested, transferred across both
domains, and corrected a diffusion ensemble under-dispersion that two algorithmic
remedies had not.

---

## 9. Reproducibility

**Code.** `src/` contains the pipeline: `data_pipeline.py` (acquisition and grid
alignment), `feature_engineering.py` (the 31 + 23 channel feature store), `deep/`
(CNN, Swin, diffusion, training loop), `deep/spectral.py` (the differentiable spectral
penalty and its evaluation accumulator, §4.8), `stacking.py`, `validation.py`,
`model_comparison.py`, `gauge_validation.py`, `publication_figures.py`,
`postprocess.py` (quantile mapping and the probability-matched mean, §4.9),
`error_spectrum.py` (§4.7), `reference_drift.py` (§2.5).

The spectral term is reached with `--spectral-weight` and `--spectral-cut` on
`src.deep.train`; `--spectral-weight 0` is byte-identical to the previous objective, and
`tests/test_deep.py` asserts that for all four loss kinds alongside the agreement with
the numpy scorer.

**Configuration.**

| file | domain / purpose |
|---|---|
| `config.yaml` | Austin, flat |
| `config_colorado.yaml` | Colorado Front Range, terrain-rich |
| `config_colorado_cool.yaml`, `config_colorado_warm.yaml` | seasonal stage-2 gate (§5.5) |
| `config_wide.yaml` | wide-domain terrain-transfer test (§5.7) |
| `config_power.yaml`, `config_power_vista.yaml` | NASA POWER input (§4.7) |

`evaluation.bbox` restricts scoring to a sub-box independently of the training extent,
which is what makes §5.7 controlled. `model.fine.months` restricts stage-2 sampling to a
season, which is what makes §5.5 possible.

**Key settings.** Coarse resolution 0.1°, fine factor 12, heavy threshold 30 mm day⁻¹,
residual-stage gate `min_gain_over_bilinear: 0.005`, checkpoint selection `rmse_pod`
with POD floor 0.633.

**Hardware.** NVIDIA GH200 (TACC Vista) and A100 (TACC Lonestar6); the tree pipeline is
CPU-only. Deep models train in 0.5–3 GPU-hours each; a full domain build from raw
acquisition to validated product takes approximately 3–5 hours, dominated by IMERG
retrieval.

**Outputs.** `results/comparison/model_comparison.{json,md}` (§4.1),
`results/gauge_validation.json` (§4.3), `results/colorado/validation_metrics.json`
(§4.5), `results/power/comparison/model_comparison.{json,csv}` (§4.7),
`results/spectral/` — weight sweep, seed repeats, 30k-step runs and both scored
comparison tables (§4.8), `results/stacking_weights.json` (§5.6),
`results/wide/validation_metrics.json` (§5.7),
`results/figures/publication/` (F1–F8, PDF + PNG).

**Data availability.** All six sources are public. IMERG requires a free NASA Earthdata
login; AORC, ERA5 and the Copernicus DEM are anonymously accessible.

---

## Appendix A. Figures

Publication figures are in `results/figures/publication/`, rendered by
`src/publication_figures.py` as vector PDF plus 600 dpi PNG at true column widths
(88 mm single, 180 mm double), Okabe–Ito palette, no in-figure titles — the caption
carries them.

| figure | width | content | section |
|---|---|---|---|
| **F1** `F1_forcing_ladder` | single | RMSE reduction by forcing regime; the 8.0 / 6.8 pair | §4.6 |
| **F2** `F2_two_domains` | double | Austin against Colorado on identical axes | §4.5 |
| **F3** `F3_accuracy_realism_plane` | double | RMSE against retained sub-10 km variance, both inputs; the spectral CNN is the point above the line | §4.2, §4.8 |
| **F4** `F4_power_spectra` | double | RAPSD of every product against AORC, both inputs, with the spectral CNN overlaid | §4.2, §4.7 |
| **F5** `F5_spectral_exchange` | double | weight sweep and training trajectories for the spectral penalty | §4.8 |
| **F6** `F6_seed_repeats` | double | family seed spread, and the within-seed paired spectral comparison | §5.8, §4.8 |
| **F7** `F7_fss_placement` | single | ΔFSS by threshold and neighbourhood width — is the added variance structure or noise | §4.8 |
| **F8** `F8_input_resolution` | double | every product under both inputs: RMSE, POD, retained variance | §4.7 |

Earlier slide-grade figures remain in `results/figures/` and
`results/figures/paper/` (P1–P8, generated by `src/figures.py`,
`src/paper_figures.py` and `src/regime_figures.py`). They are superseded for
publication by F1–F8 but still cover material without a publication counterpart:

| figure | content |
|---|---|
| `07_capacity_architecture_ablation` | §5.1 |
| `06_loss_space_ablation` | §5.3 |
| `08_extremes_remedies_ablation` | §5.4 |
| `09_overfitting_curves` | effective sample size |
| `10_diffusion_calibration`, `11_diffusion_parameterisation_fix` | §4.4 |
| `17_feature_importance_stage1`, `18_feature_importance_stage2` | §3.1 |
| `21_fields_zoom_texture` | 70 km window, convective day — the texture comparison |
| `27_subdaily_concept`, `28_subdaily_gain` | §3.2, §5.2 |
| `explainer/E1`–`E5` | problem statement, double penalty, FSS, spectral ratio, evaluation levels |

## Appendix B. Metric definitions

See [METRICS.md](METRICS.md) for formulas, units, ranges and the acronym glossary.
