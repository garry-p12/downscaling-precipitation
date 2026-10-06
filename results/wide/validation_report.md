# Validation report

Period: 2019-01-01 .. 2020-12-31 (731 days)

| Metric | ML downscaled | Bilinear baseline | IMERG orig (nearest) |
|---|---|---|---|
| rmse | 4.609 | 4.917 | 4.994 |
| bias | -0.045 | -0.082 | -0.080 |
| pearson_r | 0.837 | 0.815 | 0.810 |
| nse | 0.699 | 0.657 | 0.647 |
| kge | 0.737 | 0.787 | 0.787 |
| mae | 1.466 | 1.506 | 1.529 |
| pod_heavy | 0.552 | 0.601 | 0.600 |
| median_cell_nse | 0.710 | 0.661 | 0.649 |
| blockiness_ratio | 1.002 | 0.998 | 999.000 |

## Success criteria

- **rmse_reduction_vs_bilinear**: 0.063 (target >= 0.20) -> FAIL
- **median_cell_nse**: 0.710 (target > 0.6) -> PASS
- **frac_cells_nse_gt_0.6**: 0.885 (target report) -> n/a
- **heavy_event_pod**: 0.552 (target >= 0.80) -> FAIL
- **no_spatial_artifacts**: 1.002 (target < 1.5) -> PASS

## By intensity class (ML)

| class | n | bias | rmse | r | nse |
|---|---|---|---|---|---|
| dry | 89180592 | 0.379 | 1.066 | 0.392 | -47.858 |
| light | 14008602 | 0.754 | 4.652 | 0.365 | -2.570 |
| moderate | 5091673 | -3.049 | 10.117 | 0.396 | -2.261 |
| heavy | 2236099 | -15.122 | 25.216 | 0.496 | -0.290 |

## Figures

- wide/spatial_plots/bias_rmse_maps.png
- wide/spatial_plots/nse_maps.png
- wide/spatial_plots/example_day_2020-12-31.png
- wide/spatial_plots/spatial_corr_quarterly.png
- wide/qq_plot.png
- wide/seasonal_metrics.png
- wide/timeseries_comparisons/timeseries_locations.png
- wide/feature_importance_stage1_10km.png
- wide/feature_importance_stage2_1km.png
