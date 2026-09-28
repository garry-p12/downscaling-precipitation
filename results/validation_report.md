# Validation report

Period: 2019-01-01 .. 2020-12-31 (731 days)

| Metric | ML downscaled | Bilinear baseline | IMERG orig (nearest) |
|---|---|---|---|
| rmse | 4.655 | 4.919 | 4.994 |
| bias | -0.032 | -0.082 | -0.080 |
| pearson_r | 0.833 | 0.815 | 0.810 |
| nse | 0.693 | 0.657 | 0.647 |
| kge | 0.737 | 0.787 | 0.787 |
| mae | 1.494 | 1.506 | 1.529 |
| pod_heavy | 0.546 | 0.601 | 0.600 |
| median_cell_nse | 0.702 | 0.661 | 0.649 |
| blockiness_ratio | 1.003 | 0.998 | 999.000 |

## Success criteria

- **rmse_reduction_vs_bilinear**: 0.054 (target >= 0.20) -> FAIL
- **median_cell_nse**: 0.702 (target > 0.6) -> PASS
- **frac_cells_nse_gt_0.6**: 0.868 (target report) -> n/a
- **heavy_event_pod**: 0.546 (target >= 0.80) -> FAIL
- **no_spatial_artifacts**: 1.003 (target < 1.5) -> PASS

## By intensity class (ML)

| class | n | bias | rmse | r | nse |
|---|---|---|---|---|---|
| dry | 89180592 | 0.395 | 1.098 | 0.379 | -50.810 |
| light | 14008602 | 0.737 | 4.759 | 0.356 | -2.737 |
| moderate | 5091673 | -2.992 | 10.204 | 0.397 | -2.317 |
| heavy | 2236099 | -15.151 | 25.372 | 0.487 | -0.306 |

## Figures

- results/spatial_plots/bias_rmse_maps.png
- results/spatial_plots/nse_maps.png
- results/spatial_plots/example_day_2020-12-31.png
- results/spatial_plots/spatial_corr_quarterly.png
- results/qq_plot.png
- results/seasonal_metrics.png
- results/timeseries_comparisons/timeseries_locations.png
- results/feature_importance_stage1_10km.png
- results/feature_importance_stage2_1km.png
