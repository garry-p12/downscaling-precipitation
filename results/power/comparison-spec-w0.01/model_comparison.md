# Model comparison — 2019-01-01 .. 2020-12-31 (731 test days)

| product | RMSE | MAE | bias | r | NSE | KGE | median cell NSE | CRPS | POD>30mm | CSI>30mm | FSS 10mm@15km | spectral ratio <10km | wet-area ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IMERG (nearest) | 6.352 | 2.026 | -0.149 | 0.656 | 0.428 | 0.533 | 0.435 | 2.026 | 0.287 | 0.238 | 0.602 | 3.937 | 1.399 |
| Bilinear | 6.303 | 2.013 | -0.149 | 0.662 | 0.437 | 0.53 | 0.443 | 2.013 | 0.288 | 0.243 | 0.606 | 0 | 1.42 |
| XGBoost 2-stage | 5.679 | 2.044 | 0.288 | 0.738 | 0.543 | 0.606 | 0.552 | 2.044 | 0.382 | 0.312 | 0.674 | 0.001 | 1.871 |
| CNN (U-Net) | 5.821 | 1.982 | 0.198 | 0.725 | 0.52 | 0.651 | 0.521 | 1.982 | 0.414 | 0.314 | 0.675 | 1.281 | 1.63 |

*spectral ratio <10 km*: predicted/observed power below a 10 km wavelength. 1.0 = realistic texture; «1 = over-smoothed (a conditional-mean field); »1 = spurious fine-scale power, either blocky 10 km artifacts (nearest) or noise. *wet-area ratio*: predicted/observed fraction of cells above 1 mm/day. *CRPS*: for the deterministic products this is identical to MAE (a point forecast's CRPS); for the diffusion model it is the ensemble CRPS, which is what a probabilistic product should be judged on.
