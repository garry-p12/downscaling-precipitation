---
marp: true
theme: default
paginate: true
title: "New experiments and results"
---

<!-- Slides covering the experiments run after SLIDES.md was written: the
     five-seed replication of the combined configuration, two new selection and
     loss variants, and the pairing error that nearly went into the text.

     Intended insertion point: these replace "Combining texture and detection"
     and the slide after it, then run on. Slide 5 (pairing) is the one to keep
     if only one slide fits -- it changes how every other number is read.

     Numbers: results/seedtest_scores.json (Austin seeds 0-2),
     results/screens_scores.json (Austin seeds 3-4 and the seed-0 variants),
     results/specfloor_seeds.json (texture floor, seeds 1-4),
     results/co_rescore_matched.json (Colorado, 3 seeds, 14.50 mm),
     results/co_rescore_30mm.json (same, 30 mm, kept as the regression check).
     731 test days per domain, all scored with src/score_variants.py. -->

## What we tested since the last round

Four questions, all on the Austin domain, all scored on the 2019–20 test set.

| | question | answer |
|---|---|---|
| **1** | Does the combined configuration hold at more seeds? | **yes** — n = 3 → 5, effect unchanged |
| **2** | Is the neighbourhood loss an alternative route to texture? | **no** — null on every axis |
| **3** | Can *checkpoint selection* buy structure the loss cannot? | **no** — looked real at n = 1, null at n = 5 |
| **4** | Does the structure result survive complex terrain? | **yes** — texture 3.3×, detection +0.032 at a matched threshold |
| **5** | Were our significance claims sound? | **no** — one pairing error, now fixed |

Questions 3 and 5 are the useful ones. Question 5 was not planned: it came out
of the control runs disagreeing with each other by more than some of our
reported effects — and it then predicted exactly how question 3 would fail.

---

## 1. The combined configuration, now at five seeds

**E** = spectral penalty below 10 km (w = 0.01) + heavy-cell weight ×3 above 30 mm.
Each seed paired against *its own* control.

| | RMSE | spectral ratio | POD | CSI |
|---|---|---|---|---|
| control (MSE) | 4.657 ± 0.016 | 0.332 ± 0.091 | 0.532 ± 0.010 | 0.444 ± 0.007 |
| + spectral only *(n=3)* | 4.675 ± 0.012 | 1.056 ± 0.080 | 0.522 ± 0.005 | 0.438 ± 0.003 |
| **+ spectral + heavy ×3** | **4.653 ± 0.006** | **1.007 ± 0.105** | **0.597 ± 0.006** | **0.465 ± 0.003** |

ΔPOD **+0.0646** (t(4) = **+19.4**) · ΔCSI **+0.0214** (t = +8.0) · ΔRMSE **−0.0035** (t = −0.4)

Two more seeds moved the detection effect from +0.063 to +0.0646. **POD rises in
all five seeds; RMSE changes sign across them** — which is what "no cost" should
look like.

---

## 2. The neighbourhood loss is a null

Multiscale MSE on average-pooled fields at 1 / 3 / 9 / 27 cells — the standard
displacement-tolerant family.

| seed 0 | RMSE | texture | POD | CSI |
|---|---|---|---|---|
| control | 4.654 | 0.287 | 0.540 | 0.445 |
| multiscale | **4.637** | 0.356 | 0.542 | 0.449 |
| E | 4.661 | 1.138 | 0.592 | 0.463 |
| multiscale + E | 4.653 | 0.996 | **0.592** | 0.463 |

Best RMSE in the set, and **nothing else moves**: ΔPOD +0.0025, texture 0.287 → 0.356.

`multiscale + E` reproduces E's POD **to four decimal places** (+0.0001). The gain
is E's alone. Pooling tolerates displacement; it does not ask for structure.

---

## 3. A texture floor on selection: real at n = 1, null at n = 5

Same loss as E. One change: keep the best-RMSE checkpoint **among those whose
spectral ratio already exceeds 0.8**.

| | ΔRMSE vs E | ΔPOD vs E |
|---|---|---|
| seed 0 — *the one we had* | **−0.0190** | **+0.0181** |
| seed 1 | −0.0046 | −0.0042 |
| seed 2 | +0.0137 | −0.0013 |
| seed 3 | 0.0000 | 0.0000 |
| seed 4 | 0.0000 | 0.0000 |
| **mean (n = 5)** | **−0.0020** (t = −0.4) | **+0.0025** (t = +0.6) |

Seed 0 was the best pair we had measured. It was **the seed lottery**.

At seeds 3–4 the floor is a *no-op*: the best-RMSE checkpoint already cleared
0.8, so it kept the same weights and the scores are identical to the digit.
The floor binds only when selection and texture disagree — and when it does,
it helps as often as it hurts.

---

## 4. Complex terrain: texture replicates, detection is poor

Colorado Front Range, 3 seeds, paired within seed. Detection at **14.50 mm** —
the depth that reproduces Austin's 30 mm *exceedance rate* (2.0 %).

| | RMSE | spectral ratio | KGE | POD | freq. bias |
|---|---|---|---|---|---|
| control | 2.978 ± 0.003 | 0.351 ± 0.019 | 0.329 ± 0.011 | 0.127 | **0.146** |
| **E** | **2.938 ± 0.020** | **1.160 ± 0.189** | **0.374 ± 0.017** | **0.159** | 0.191 |

**Texture replicates: 0.35 → 1.16, a factor of 3.3** against Austin's 3.0 — no
longer a flat-terrain result. RMSE improves in all 3 seeds (−0.040), where Austin
was null. ΔPOD **+0.032** (t = 4.35), ΔCSI **+0.029** — half the Austin effect,
same direction.

**Why the threshold had to change:** at 30 mm the control scores POD 0.013 at a
frequency bias of **0.039**. 30 mm is **6.7× rarer** here than over Austin.

> **But the fix does not rescue the regime.** At matched exceedance the frequency
> bias is still **0.15**, against Austin's **0.73**. Both models under-forecast
> heavy rain in terrain by ~5×. The objective *helps* where detection is poor;
> it does not make detection good.

---

## 5. The pairing error, and why it matters

We scored several variants against **one** control run. Then we scored two
*controls* against each other:

> `ctl_s4` vs `ctl_s3`: ΔRMSE **−0.043**, ΔPOD **+0.016** — both with
> day-block bootstrap CIs **excluding zero**.

Both runs are the identical configuration. The only difference is the seed.

| | control spread across 5 seeds |
|---|---|
| RMSE | 4.6380 – 4.6814 (sd **0.0163**) |
| POD | 0.5189 – 0.5421 (sd **0.0097**) |

**A single-control bootstrap manufactures significance.** The interval describes
sampling over *days*; it knows nothing about sampling over *initialisations*.

**Fix:** pair within seed, and read every effect against the control spread.
E's ΔPOD is 6.6× the control POD sd — which is why it survives. Its ΔRMSE is a
fifth of the control RMSE sd — which is why "no cost" is the honest reading
rather than "a small gain".

---

## Where this leaves the menu

| if you need… | use | evidence |
|---|---|---|
| lowest error | XGBoost (4.583) | — |
| structure *and* detection | **spectral + heavy ×3** | 5 seeds Austin, 3 seeds Colorado |
| displacement tolerance | *not* the multiscale loss | null, 1 seed |
| texture-floored selection | *not worth it* | **null, 5 seeds** |

**One survivor, two casualties.** Of the three interventions tested this round,
only the one we already had replicated. Both new ideas died — and the pairing
discipline from slide 5 is what killed the one that had looked best.

**The result that outlives the numbers:** *seed spread is the yardstick.* Any
effect smaller than the control's seed-to-seed spread is not an effect, whatever
the bootstrap says. We proved this on ourselves: the texture floor's +0.018 POD
at seed 0 was inside a ±0.0097 control spread, and four more seeds took it to
+0.0025.

---

## What is still open

| question | why it is not answered |
|---|---|
| Why do terrain models under-forecast heavy rain **5×**? | freq. bias 0.15 vs Austin 0.73, at matched exceedance |
| Does E's **RMSE gain** in Colorado hold? | 3 seeds, all negative, t = −3.1 vs crit 4.30 |
| Is the texture gain worth anything **operationally**? | no decision metric tested, only scores |

**One domain caveat remains.** Austin has **0.000 cells above 5° slope**;
Colorado has 0.255. The texture effect now has both, at ~3× each, and the
detection effect has both once the threshold is matched on exceedance rather
than on millimetres.
