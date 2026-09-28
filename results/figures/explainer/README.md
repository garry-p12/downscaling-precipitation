# Explainer figures

Conceptual figures for [`docs/METRICS.md`](../../../docs/METRICS.md). They use small synthetic fields, because the mechanisms are far clearer in a controlled example than in a real rainfall day. All quoted numbers are computed when the figure is rendered.

Regenerate with `python -m src.explainer_figures`.

| figure | what it explains |
|---|---|
| [`E1_the_problem.png`](E1_the_problem.png) | One 10 km cell covers a 12 x 12 block of 1 km cells (outlined). The model must produce 144 values whose mean is fixed by the satellite. Nothing in the inputs says which square kilometre got the storm core, which is the constraint every metric below is probing. |
| [`E2_double_penalty.png`](E2_double_penalty.png) | Both candidates carry the same total water. The smooth field scores RMSE 3.54 while keeping only 20% of the true peak; the sharp field displaced by 8 km scores 5.85 - 1.7x worse - despite having the right intensity. A displaced feature is punished twice (missed where it was, false alarm where it was not), so minimising RMSE drives models towards fields that cannot occur in nature. |
| [`E3_fss.png`](E3_fss.png) | A field displaced by 6 cells scores near zero at grid-point level but recovers as the neighbourhood widens - it had the right structure in roughly the right place. An over-smoothed field does not recover, because the structure is gone at every scale. RMSE cannot make this distinction. |
| [`E4_spectral_ratio.png`](E4_spectral_ratio.png) | Smoothing removes power at short wavelengths (ratio 0.00); adding noise adds too much (ratio 319769839.73). The diagnostic is two-sided, so a product far above 1.0 is as wrong as one far below - it is texture in the wrong places, not skill. |
| [`E5_two_levels.png`](E5_two_levels.png) | Identical prediction, two scoring levels: RMSE 1.47 at 1 km against 0.45 at 10 km (70% lower) purely because averaging 144 cells cancels independent error. A 10 km score always flatters a product, so the project's 10 km and 1 km numbers must never be compared with each other - only within a level. |
