# Publication-grade figures

Test period 2019-01-01 to 2020-12-31 (731 days), 1 km, against AORC. Rendered at 300 dpi with a PDF alongside each PNG.

Palette is Okabe-Ito (colour-blind safe, legible in greyscale). Regenerate with `python -m src.paper_figures`.

| figure | caption |
|---|---|
| [`P1_headline_scores.png`](P1_headline_scores.png) | Deterministic and probabilistic skill of every product over the 2019-2020 test period (731 days, 1 km, against AORC). Dashed line marks bilinear interpolation; axes are zoomed to the range spanned by the products, which is why dots rather than bars are used. All learned products beat bilinear on RMSE; only the CNN and the transformer also beat it on heavy-event detection, and only the diffusion ensemble improves CRPS. |
| [`P2_accuracy_realism_tradeoff.png`](P2_accuracy_realism_tradeoff.png) | Squared-error skill against retained fine-scale variance; the grey line is the Pareto front (RMSE axis inverted, so upper-right is better on both). Products that score best on RMSE retain 2-20 % of the observed sub-10 km variance, while the only product with realistic texture has the worst RMSE. This is the double penalty made quantitative. |
| [`P3_taylor_diagram.png`](P3_taylor_diagram.png) | Correlation (azimuth), standard deviation normalised by the observed value (radius) and centred RMS difference (dotted arcs about the star) for every product. A product on the unit radius has the right variability; all learned products fall inside it, i.e. they are smoother than the observations. |
| [`P4_power_spectra.png`](P4_power_spectra.png) | Radially averaged power spectral density. The shaded band is the sub-10 km range summarised by the spectral ratio. Deterministic products lose one to two orders of magnitude of variance below ~30 km; nearest-neighbour exceeds the observations because 10 km blocks inject spurious high-frequency power. |
| [`P5_fractions_skill_score.png`](P5_fractions_skill_score.png) | Fractions Skill Score against neighbourhood width for three intensity thresholds. FSS credits rain placed approximately correctly, so it separates products that lose structure from those that merely displace it; the products converge at large neighbourhoods and at low thresholds. |
