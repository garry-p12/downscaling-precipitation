# Downscaling NASA POWER precipitation from 0.5° to 1 km

What we did, what we found, and what the numbers mean. Everything is measured on
2019–2020 over a central Texas domain, a period no model trained on.

Figures are in `results/figures/review_power/`.

---

## 1. The problem

NASA POWER serves `PRECTOTCORR`, a daily precipitation estimate drawn from the MERRA-2
reanalysis — a physics model with observations assimilated into it — and bias-corrected
against CPC gauge data. It is available globally, from 1981, with no gaps. That makes it
attractive for anywhere and any period a satellite record does not cover.

It is also **coarse**. A single POWER cell is 0.5° × 0.625°, roughly 55 km by 55 km. Our
target grid is 1 km, so **one POWER value has to be turned into about 3,600 of them.**
The evaluation domain holds just 42 POWER cells.

We take that coarse field and try to produce a 1 km map, checking against **AORC**, a
1 km national rainfall analysis we treat as the truth.

**Time is split so nothing leaks.** Training ends in 2017, 2018 is held out for tuning,
and every number reported here is from **2019–2020**, which no model saw.

## 2. What we built

Five approaches, each testing a different idea about where the missing detail comes
from:

- **Bilinear interpolation** — smooth the coarse field out. The thing to beat. Every
  model is constructed as *bilinear + a correction*, so an untrained model is exactly
  this baseline and no reported gain can be an artefact of tuning.
- **XGBoost** — thousands of small decision rules, one prediction per cell.
- **CNN (U-Net)** — looks at neighbourhoods rather than single cells.
- **Swin transformer** — can see far across the map, for cases where what matters is
  upwind.
- **Diffusion** — samples from the range of plausible fields instead of predicting one.
- **CNN + texture penalty** — a CNN trained with an added penalty for producing a field
  that is too smooth, plus extra weight on heavy-rain cells.

## 3. How they score

![metric comparison](../results/figures/review_power/R1_metric_comparison.png)

| | error (mm/day) | heavy rain found | CSI | KGE | texture |
|---|---|---|---|---|---|
| bilinear | 6.303 | 0.288 | 0.243 | 0.530 | 0.000 |
| **XGBoost** | **5.679** | 0.382 | 0.312 | 0.606 | 0.001 |
| CNN + texture penalty | 5.821 | 0.414 | 0.314 | **0.651** | **1.121** |
| Swin | 5.886 | 0.402 | 0.309 | 0.636 | 0.058 |
| Diffusion | 5.947 | 0.394 | 0.302 | 0.635 | 0.185 |
| CNN | 6.078 | **0.509** | **0.342** | 0.617 | 0.250 |

Every model beats interpolation. XGBoost takes the error from 6.303 to **5.679**, a
**10 % reduction**, and the CNN finds **51 %** of heavy-rain cells against interpolation's
29 %.

**"Texture" is how much fine-scale detail a field has compared with real rain**, where
1.0 is correct. Most of these products are far below it — a field that is far too
smooth. The exception is the CNN with the texture penalty at **1.121**, essentially
right.

## 4. What the products get wrong

### They rain too often, and not hard enough

![intensity distribution](../results/figures/review_power/R2_intensity_distribution.png)

Real rain falls on **28 %** of cells. These products say **48–62 %** — they spread a
little rain over roughly twice too much ground.

![tail shortfall](../results/figures/review_power/R9_quantile_ratio.png)

At the same time they refuse to produce the very heaviest values. At the 1-in-10,000
cell they reach only **48–72 %** of what actually fell. Diffusion carries the tail
furthest at 72 %, which is what sampling a distribution rather than averaging one is
for; XGBoost is worst at 48 %.

Both failures come from the same place. Training rewards being close *on average*, and
the safest way to be close on average is to hedge — put a little rain everywhere and
never commit to a large number.

### The heavier the storm, the less of it survives

![storm events](../results/figures/review_power/R3_storm_events.png)

We scored six real storms from the test period individually. Across them the products
recover roughly a fifth to two-fifths of the observed peak, and the shortfall grows
with the size of the event.

### An independent check confirms it

![gauge check](../results/figures/review_power/R5_gauge_events.png)

AORC is itself an analysis, so agreement with it is not proof. We also compared against
**686 GHCN-D rain gauges** inside the domain — measurements that do not come from the
same construction.

On 5 June 2019 the gauges recorded a peak of **305 mm**:

| | peak produced | fraction of the gauge peak |
|---|---|---|
| AORC | 311 mm | **1.02** |
| CNN | 129 mm | 0.42 |
| Diffusion | 123 mm | 0.40 |
| Swin | 113 mm | 0.37 |
| CNN + texture | 112 mm | 0.37 |
| XGBoost | 79 mm | 0.26 |

AORC matches the gauges almost exactly. The products reach a quarter to two-fifths. **The
peaks are genuinely missing** — this is not a disagreement with our reference.

## 5. Where the error actually is

This is the central result, and it changes what the other numbers mean.

Each product's error can be split into two parts: the error already present in its
**0.5° field**, and the extra error added by going from 0.5° to 1 km.

![error budget](../results/figures/review_power/R8_error_budget.png)

| | total | in the 0.5° field | added by the 0.5° → 1 km step |
|---|---|---|---|
| bilinear | 6.303 | 5.392 | 3.263 |
| XGBoost | 5.679 | **4.655** | 3.253 |
| CNN | 6.078 | 5.132 | 3.256 |
| Swin | 5.886 | 4.906 | 3.251 |
| CNN + texture | 5.821 | 4.825 | 3.256 |

**Between two thirds and three quarters of every product's error is already in its
coarse field** before any sharpening happens.

And look at the last column. Five very different models — decision trees, a
convolutional network, a transformer, a diffusion model — produce sharpening errors
spanning **3.251 to 3.379**, a range of 0.13. Their coarse errors span 0.74, nearly six
times as much. **The models differ in how well they correct the coarse field, not in how
well they sharpen it.**

## 6. Is the sharpening doing anything at all?

A direct test. Take XGBoost's own corrected 0.5° field and simply hold each value flat
across its block — no sharpening whatsoever — then score that against the full 1 km
output:

| scored at 1 km | error |
|---|---|
| XGBoost, full 1 km output | **5.679** |
| XGBoost's 0.5° field, held flat in blocks | 5.729 |

**The entire 0.5° → 1 km step is worth 0.051 mm.** Splitting the gain over interpolation
into its two sources: **99 % is correcting the coarse field, 1 % is sharpening.**

Put another way: interpolation leaves **3.26 mm** of error that a sharpening step could
in principle reach. The model removes **0.3 %** of it.

So what this system is, measured rather than assumed, is a **good 0.5° bias corrector
that then interpolates**. The product is genuinely better than the alternatives — 10 %
better than interpolation on unseen years — but the improvement comes from fixing the
coarse values, not from adding 1 km detail.

## 7. The ceiling

How much of what remains is even recoverable? To find out we replaced POWER with a
**perfect** 0.5° input — the truth itself, averaged down to 0.5°. Everything else
identical. Any error left is only what the averaging destroyed.

| input | error |
|---|---|
| real POWER, best model | 5.679 |
| **perfect 0.5° input** | **3.001** |

**No model fed a 0.5° input can do better than about 3.0 mm on this domain**, however
good the model is. That is the floor set by the grid itself.

It also says where the remaining error lives: of the 5.68 we have, about 3.0 is the grid
spacing and the rest is POWER being wrong. The larger share is the accuracy of the
coarse values, not their spacing.

## 8. What this means in practice

**Useful for:** rainfall totals over a catchment, a season, or a region. The products
beat interpolation by 10 % on years they never saw, and for a global, gap-free,
1981-onwards record that is a real capability.

**Not usable for:** the peak of a severe storm. The products recover a quarter to
two-fifths of observed peaks, confirmed against independent gauges, so a design-storm or
flash-flood application would be badly misled.

**If you need a realistic-looking field** — input to a hydrology model, or any use where
an over-smooth map misleads — the CNN with the texture penalty is the product. It is the
only one that reaches approximately correct fine-scale structure, and it does so without
costing accuracy.

**One caveat to carry.** POWER's precipitation is bias-corrected against CPC gauges, and
AORC also assimilates gauge data. The two therefore share information, so POWER's
absolute skill against AORC should not be read as fully independent validation. This
does not affect the error split in §5 or §6, both of which compare a product against
itself.

## 9. Where the next gain is

The measurements point one way. More capacity will not help: five model families with
very different inductive biases produce sharpening errors within 0.13 mm of each other.
A better sharpener cannot help much either: the whole step is worth 0.05 mm and a
perfect one would be worth 3.26.

**The gain is in the 0.5° field**, which carries two thirds to three quarters of the
error. That means giving the coarse stage better information — independent precipitation
estimates with different error characteristics, or predictors that flag where the
reanalysis is least reliable — rather than building a cleverer way of sharpening a field
that is already wrong when it arrives.
