# Result figures

Domain: Austin, TX (`[-99.0, 28.5, -96.0, 32.0]`). Test period 2019-01-01 to 2020-12-31 (731 days) against AORC at 1 km.

Regenerate with `python -m src.figures`.

| figure | what it shows |
|---|---|
| [`01_test_metrics_comparison.png`](01_test_metrics_comparison.png) | Headline deterministic and probabilistic scores for every product on the 2019-2020 test years. Lower is better except NSE and POD. RMSE spans only 4.86-5.15 across four model classes, while CRPS separates them clearly - the diffusion ensemble wins the probabilistic score. |
| [`02_power_spectra.png`](02_power_spectra.png) | Radially averaged power spectra on the 120 wettest test days. All deterministic products fall 1-2 orders of magnitude below AORC below ~30 km; IMERG-nearest sits above it (blocky 10 km steps). |
| [`03_spectral_ratio.png`](03_spectral_ratio.png) | Every product carries 2-13% of the observed sub-10 km variance. Minimising squared error returns the conditional mean, which for convective rain is genuinely smooth - no architecture fixes this. |
| [`04_skill_by_intensity.png`](04_skill_by_intensity.png) | Scores split by observed rainfall class. The heavy class carries a large negative bias (-14 mm for the tree model): all products shrink extremes toward the mean. |
| [`05_detection_by_threshold.png`](05_detection_by_threshold.png) | Hit rate, false alarms and CSI at 1/10/30 mm. Every learned product loses to plain bilinear interpolation at the 30 mm threshold - the core extremes failure. |
| [`06_loss_space_ablation.png`](06_loss_space_ablation.png) | Training in log space but scoring in mm costs 0.43 mm RMSE and drives bias to -1.4 mm: log-space MSE fits the mean of log precipitation, which under-predicts the mean in mm. |
| [`07_capacity_architecture_ablation.png`](07_capacity_architecture_ablation.png) | A 1.65 M CNN, an 11.7 M CNN and a 7.9 M transformer converge to the same dev RMSE and all overfit after ~500 steps. The binding constraint is ~1100 independent weather days, not capacity. |
| [`08_extremes_remedies_ablation.png`](08_extremes_remedies_ablation.png) | Six objectives at their best checkpoint. Heavy-event sample weighting is the only change that moves POD (0.612 -> 0.633, matching bilinear) and cuts bias 3.4x, at an RMSE cost of 0.0014. The quantile losses do almost nothing because the best checkpoint arrives at step 500. |
| [`09_overfitting_curves.png`](09_overfitting_curves.png) | Dev skill against training step. RMSE is best at step 500-2000 and worsens thereafter, which is why loss-shape changes (quantile) never take effect: training stops before they can act. |
| [`10_diffusion_calibration.png`](10_diffusion_calibration.png) | After the x0 fix the ensemble recovers only ~56% of the observed sub-grid spread at every noise level, leaving the field ~2 mm/day too dry. Under-dispersed, i.e. over-confident: an ensemble spanning half the true uncertainty would under-warn at a flood threshold. |
| [`11_diffusion_parameterisation_fix.png`](11_diffusion_parameterisation_fix.png) | The cosine schedule drives abar to 1e-8 at t=T, so recovering the residual divides by ~1e-4 and amplifies error 10,000x: sampled spread ran 87% too wide and the field was +4.5 mm biased. Predicting x0 divides by sqrt(1-abar) ~ 1 instead, leaving the opposite, milder failure (spread 56% of observed). |
| [`12_fractions_skill_score.png`](12_fractions_skill_score.png) | Fractions Skill Score against neighbourhood size at 1/10/30 mm - credit for placing rain approximately correctly rather than exactly. |
| [`13_spatial_bias_rmse_maps.png`](13_spatial_bias_rmse_maps.png) | Mean field, per-cell bias and per-cell RMSE for the tree product and both baselines, 2019-2020. |
| [`14_per_cell_nse_maps.png`](14_per_cell_nse_maps.png) | Per-cell Nash-Sutcliffe efficiency. Median cell NSE is 0.68 for the tree product. |
| [`15_quantile_quantile.png`](15_quantile_quantile.png) | Quantile-quantile against AORC on 200k sampled test cells; departures at the top end are the extreme-value underprediction. |
| [`16_city_timeseries.png`](16_city_timeseries.png) | Daily series at seven towns across the domain for the test years. |
| [`17_feature_importance_stage1.png`](17_feature_importance_stage1.png) | Gain importance for the 10 km tree stage. IMERG itself and its neighbourhood mean dominate; the climatological percentile ranks third. |
| [`18_feature_importance_stage2.png`](18_feature_importance_stage2.png) | Gain importance for the 1 km residual stage - the stage that was ultimately rejected on the hold-out. |
| [`19_example_wettest_day.png`](19_example_wettest_day.png) | The wettest test day at 1 km for every product against AORC - the visual counterpart to the spectra: deterministic fields are smooth blobs, AORC has structure. |
| [`20_fields_heavy_day_domain.png`](20_fields_heavy_day_domain.png) | Every 1 km product against AORC for 2020-12-31, the wettest day of the test period. Same colour scale throughout. The learned products track the large-scale pattern but render it as smooth blobs where AORC has sharp cores. |
| [`21_fields_zoom_texture.png`](21_fields_zoom_texture.png) | A ~70 km window on 2020-09-02, centred where AORC has the most small-scale structure. At this scale the difference between the products is obvious: AORC resolves a sharp rain band, IMERG-nearest renders 10 km blocks, and the deterministic products (bilinear, XGBoost, CNN, Swin) smooth the band away entirely. Only the single diffusion member carries comparable fine-scale structure. This is the spectral-ratio figure made visible. |
| [`22_field_errors_heavy_day.png`](22_field_errors_heavy_day.png) | Product minus AORC on 2020-12-31. Errors are dipoles around the observed cores - the classic double penalty: a smooth field misses the peak and spills rain around it. |
| [`23_mean_fields.png`](23_mean_fields.png) | Two-year mean per product. Long averages hide the smoothing problem - every product reproduces the climatological gradient, which is why daily fields and spectra are the honest diagnostic. |
| [`24_fields_three_days.png`](24_fields_three_days.png) | A heavy, a moderate and a light day, each row on its own colour scale so the light day is not washed out. On the two low-intensity rows AORC shows radial spokes and banding - these are radar artifacts in the reference dataset itself, not model error, and they are a reminder that AORC is an analysis rather than truth. Every product inherits the large-scale pattern; the sharp cores are lost in all of them. |
| [`25_diffusion_ensemble_vs_member.png`](25_diffusion_ensemble_vs_member.png) | Averaging members smooths the field back toward the conditional mean, so the ensemble mean scores better on RMSE while the single member is the one meant to look like rain. The member does carry visible fine-scale structure the deterministic products lack - but it places it imprecisely, and domain-wide the ensemble still spans only 56% of the observed spread. |
| [`26_checkpoint_selection.png`](26_checkpoint_selection.png) | The CNN's heavy-event POD rises from 0.616 to 0.742 over training while dev RMSE passes through a minimum at step 5000 and then drifts up by 5%. Here both rules land on step 5000 (POD 0.694): with 6423 training days the model clears the POD floor well before its RMSE optimum, so the constraint is slack. On the 1096-day record the same rule changed the outcome - the model only reached high POD long after RMSE had started degrading. More data relaxes the trade-off rather than removing it. |

## Companion figure sets

* [`explainer/`](explainer/) - conceptual figures for `docs/METRICS.md`: what the metrics measure and why RMSE alone misleads on this task.
* [`paper/`](paper/) - publication-grade versions (300 dpi + PDF, Okabe-Ito palette, panel labels), including a Taylor diagram and the accuracy-realism trade-off.

## Reading order

1. **01, 03, 04, 05** - what the finished products do on the test years.
2. **02, 12** - whether the fields look like rain (spectra, FSS) rather than merely scoring well.
3. **06-09** - the training decisions that got there, each judged on the 2018 dev year only.
4. **10, 11** - the diffusion model's two failure modes and the fix between them.
5. **13-19** - spatial, per-location and single-day detail.

Figures 13-18 are the validation plots for the tree product (the `ml` method in
`results/validation_metrics.json`); 01-12 and 19 score every product together.
