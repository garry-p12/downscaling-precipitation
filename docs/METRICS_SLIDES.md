---
marp: true
theme: default
paginate: true
title: "Metric slides"
---

<!-- Three slides to drop into SLIDES.md. Slide 1 is the one to use; 2 and 3 are
     backup. Intended insertion point: after "Evaluation — no single number is
     sufficient". Full discussion stays in METRICS.md. -->

## The metric suite

| | metric | what it asks | watch out for |
|---|---|---|---|
| **Accuracy** | RMSE | typical error, squaring big misses | **rewards smoothing** |
| | MAE | typical error on an ordinary day | blind to extremes |
| | bias | systematically too wet or dry | can be ~0 while every cell is wrong |
| | NSE | better than predicting the long-run mean? | 1 perfect, 0 worthless, <0 worse than a constant |
| | KGE | right amount *and* variability *and* pattern | unlike RMSE, **penalises over-smoothing** |
| **Realism** | spectral ratio <10 km | what share of observed fine-scale variance survives? | **1.0 = realistic**; ≫1 = noise |
| | FSS | is rain *approximately* in the right place? | scored over a neighbourhood, so small displacement isn't fatal |
| | wet-area ratio | is the right fraction of the map raining? | >1 = drizzling everywhere |
| **Extremes** | POD >30 mm | of cells that really had ≥30 mm, how many did we flag? | **inflated by over-forecasting** |
| | FAR | of cells we flagged, how many were wrong? | must be read with POD |
| | CSI | hits, penalised for misses *and* false alarms | the honest single detection number |
| | frequency bias | do we forecast heavy rain as *often* as it happens? | 1.0 = yes; 0.74 = a conditional mean |
| **Uncertainty** | CRPS | accurate *and* honestly uncertain? | collapses to MAE for one raster, so ensembles compare |
| | ensemble spread | does it span the real uncertainty? | 0.59 = over-confident by ~40 % |

---

## How to read a comparison table

1. **RMSE and bias** — is the magnitude right?
2. **KGE and spectral ratio** — or the RMSE winner may simply be the blurriest
3. **POD with FAR, CSI and frequency bias** — POD alone rewards over-forecasting
4. **FSS at several widths** — is it roughly in the right place?
5. **CRPS and spread together** — good CRPS at 0.59 spread is an over-confident forecast

> **A product is only genuinely better if it wins in more than one family.**

Two evaluation levels are never compared: a 1 km score is always worse than a
10 km score on the same product, because averaging 144 cells cancels independent
error.

If a product was trained with a spectral term, its spectral ratio is an
*optimised* quantity — no longer an independent check on it.

---

## Formulas *(backup)*

| acronym | full form | formula |
|---|---|---|
| RMSE | Root Mean Squared Error | `sqrt(mean((pred − obs)²))` |
| NSE | Nash–Sutcliffe Efficiency | `1 − MSE / var(obs)` |
| KGE | Kling–Gupta Efficiency | `1 − sqrt((r−1)² + (σp/σo − 1)² + (μp/μo − 1)²)` |
| CRPS | Continuous Ranked Probability Score | `mean│X−y│ − ½ mean│X−X′│` |
| POD | Probability of Detection | `hits / (hits + misses)` |
| FAR | False Alarm Ratio | `FA / (hits + FA)` |
| CSI | Critical Success Index | `hits / (hits + misses + FA)` |
| FSS | Fractions Skill Score | `1 − mean((Pf−Po)²) / (mean(Pf²) + mean(Po²))` |
| RAPSD | Radially Averaged Power Spectral Density | variance by spatial wavelength |

Baselines at 1 km: bilinear **4.919**, nearest **4.994**. Every network outputs
`bilinear(coarse) + correction` with a zero-initialised head, so an untrained
model *is* the bilinear baseline.
