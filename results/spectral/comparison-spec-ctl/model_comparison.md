# Model comparison — 2019-01-01 .. 2020-12-31 (731 test days)

| product | RMSE | MAE | bias | r | NSE | KGE | median cell NSE | CRPS | POD>30mm | CSI>30mm | FSS 10mm@15km | spectral ratio <10km | wet-area ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IMERG (nearest) | 4.994 | 1.529 | -0.08 | 0.81 | 0.647 | 0.787 | 0.649 | 1.529 | 0.6 | 0.44 | 0.759 | 32.71 | 0.997 |
| Bilinear | 4.919 | 1.506 | -0.082 | 0.815 | 0.657 | 0.787 | 0.661 | 1.506 | 0.601 | 0.445 | 0.757 | 0.125 | 1.011 |
| XGBoost 2-stage | 4.583 | 1.467 | -0.046 | 0.839 | 0.702 | 0.739 | 0.711 | 1.467 | 0.548 | 0.458 | 0.769 | 0.021 | 1.32 |
| CNN (U-Net) | 4.654 | 1.492 | -0.088 | 0.834 | 0.693 | 0.732 | 0.697 | 1.492 | 0.54 | 0.445 | 0.759 | 0.303 | 1.252 |

*spectral ratio <10 km*: predicted/observed power below a 10 km wavelength. 1.0 = realistic texture; «1 = over-smoothed (a conditional-mean field); »1 = spurious fine-scale power, either blocky 10 km artifacts (nearest) or noise. *wet-area ratio*: predicted/observed fraction of cells above 1 mm/day. *CRPS*: for the deterministic products this is identical to MAE (a point forecast's CRPS); for the diffusion model it is the ensemble CRPS, which is what a probabilistic product should be judged on.
