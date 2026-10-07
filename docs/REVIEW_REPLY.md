# Draft reply

> Draft for Guruprasad to edit and send. Figures referenced are in
> `results/figures/review/` (Austin) and `results/figures/review_co/` (Colorado).

---

Thank you — these were the right three things to ask for, and the third one in
particular changed what I think the honest conclusion is.

One correction before the substance: the second domain is the **Colorado Front
Range**, not California. It matters for the argument, because the two domains were
built to be identical in size (1,050 coarse cells, 151,200 fine) and to differ only
in forcing — Austin is flat and convective (0.000 cells above 5° slope), Colorado is
orographic (0.255).

## A. Why these five methods

I have added the rationale to §3.3 of the paper, with a reference list. The short
version is that the set is not a survey — each family is one representative of a
different hypothesis about *where* 1 km structure comes from, so a negative result
tells us which hypothesis failed rather than which implementation was weaker:

- **Bilinear** is the null. Every model is parameterised as `bilinear(coarse) +
  correction` with a zero-initialised head, so an untrained network *is* the
  baseline and no reported gain can come from tuning one side harder. This is the
  residual framing standard in super-resolution since Kim et al. (2016).
- **XGBoost (2-stage)** tests "it is a point-wise bias problem". Boosted trees remain
  the strong baseline on tabular data (Chen & Guestrin 2016; Grinsztajn et al. 2022),
  and the two-stage split mirrors the classical bias-correction / disaggregation
  decomposition so the two steps can be judged separately.
- **CNN (U-Net)** tests "it is a spatial pattern problem" — Ronneberger et al. (2015),
  with DeepSD (Vandal et al. 2017) as the precipitation-downscaling reference point.
- **Swin** tests "it needs long-range context", since orographic rainfall depends on
  flow interacting with terrain tens of kilometres upstream and a convolution has a
  bounded receptive field (Liu et al. 2021; Liang et al. 2021).
- **Diffusion** tests "the conditional mean is the wrong estimator". Every model above
  minimises a per-cell loss, which is minimised by a field smoother than real rain —
  a property of the objective, not the architecture, so capacity cannot fix it.
  Residual diffusion samples the distribution instead (CorrDiff, Mardani et al. 2025).

## B. Figures

Five, in the journal house style, for each domain:

| | |
|---|---|
| **R1** | RMSE, POD, CSI and sub-10 km power ratio per product, with seed spread as error bars |
| **R2** | intensity PDF and exceedance curve against AORC |
| **R3** | per-storm peak capture |
| **R4** | RMSE and POD by season |
| **R5** | gauge verification, per storm and overall |

The error bars on R1 are deliberate: several differences between models are inside the
seed-to-seed spread and should not be read as a ranking.

## C. Storms, distributions, seasons and gauges

All of this is on **2019–2020, which is outside the training record** (training ends
2017-12-31, 2018 is held out as a dev year for every model family). Events were chosen
from the data two ways, because they fail differently: wettest by area mean, and
wettest by point maximum. Colorado's set includes **2019-03-13, the Front Range bomb
cyclone**.

**1. Peak capture collapses as events get larger.** Austin, ratio of predicted to
observed event peak: about 0.8 for a 105 mm event, 0.65 at 160–175 mm, and 0.20–0.32
at 270–285 mm. On 2020-05-12 the observed peak was 284 mm and the best product reached
64 mm. Colorado is worse: 0.6 at 65 mm falling to 0.1 at 130 mm.

**2. The gauges confirm this is real, not an artefact of scoring against AORC.** Across
687 GHCN-D stations in Austin, AORC's event peak is 1.02–1.06 of the gauge peak on five
of six storms, while the products sit at 0.19–0.53. On 2019-06-05 the gauges recorded
305 mm, AORC 311 mm, and the best product 161 mm. Observation-hour alignment was chosen
per station on AORC and then applied unchanged to every product so it could not favour
one of them (571 of 687 stations align at zero days).

**3. The distributions are wrong in both tails.** Wet fraction is roughly twice observed
— 0.57–0.60 against AORC's 0.280 in Austin, 0.68–0.69 against 0.395 in Colorado — so the
products drizzle over far too much area. At the same time the upper tail is short:
66–83 % of AORC's 99.99th percentile in Austin, 45–63 % in Colorado. RMSE rewards both
errors, which is exactly why the figures were worth asking for.

**4. Diffusion is the one family that partly escapes this**, which is what its design
predicts: it carries the tail furthest (83 % of AORC's 99.99th percentile against
XGBoost's 66 %) and has the best peak ratio on most localised events. It pays for that
in mean RMSE.

**5. A caveat on gauges in terrain.** The Colorado gauge comparison inverts — AORC scores
*worse* against gauges than the models do (median station RMSE 3.13 against 2.90–2.95).
I do not think that means the models are better there. Front Range gauges sit in valleys
and towns while orographic maxima sit on slopes, so a smoother field matches a sparse
valley sample better than a sharp one. AORC's event peak ratios are correspondingly
erratic, including 15.3 on 2019-05-31 where the gauges averaged 0.2 mm and AORC put
89 mm where no gauge stood. 77 % of Colorado stations correlate below r = 0.7 with AORC
at their own cell, against 62 % in Austin. I would treat gauge verification as materially
weaker evidence in complex terrain and lean on the Austin result.

## On whether the downscaling is useful

The most useful thing I can report is a result from a control experiment I ran
alongside these. If the 10 km input is replaced with AORC itself averaged to 10 km —
a perfect input by construction, same pipeline otherwise — RMSE falls from 4.58 to
**0.72** in Austin and from 2.98 to **0.49** in Colorado. Decomposed, the 10 km → 1 km
step carries about **2.5 % of the total error**; the rest is the satellite being wrong
at 10 km.

So the honest reading is:

- For **areal totals over a catchment**, the products are useful and clearly beat
  interpolation.
- For **localised extremes — the design storm case — none of them is usable yet.** They
  recover a fifth to a third of the observed peak, and the gauges agree.
- The ceiling is set by **input accuracy, not resolution**. A perfect 0.5° input still
  beats our best real 0.1° product by 1.6 mm, which says the retrieval error dominates
  the resolution penalty.

That last point has redirected the work: the gains now come from giving the 10 km stage
better information rather than from better downscaling. Adding ERA5 reanalysis
precipitation as a predictor cut the 10 km error by 0.15 mm in Austin and 0.27 mm in
Colorado — roughly twice as much in the orographic domain, as expected, since that is
where the microwave retrieval is weakest.

Happy to walk through any of the figures, and glad to add further storm cases if there
are particular events you would like to see.
