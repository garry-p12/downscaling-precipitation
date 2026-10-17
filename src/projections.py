"""Bias-correct NEX-GDDP-CMIP6 precipitation against the observed record.

Stage one of the RSIR proposal's projections arm. The proposal describes it as
a supervised mapping:

    "learning a mapping function F such that historical simulations from CMIP6
     or NASA-NEX-CMIP6 models (input X) are translated into high-resolution
     NASA-POWER-like outputs (target y) ... Once this model is trained and
     validated using historical periods where both x and y are available"

**That cannot be done day by day, and the reason is not a technical detail.**
A CMIP6 historical run is a free-running climate simulation. It is initialised
from a pre-industrial spin-up and evolves its own weather; its 3 June 2010 has
no relationship to the real 3 June 2010 beyond sharing a season and a forcing
level. Pairing days and regressing would be fitting noise, and the fit would
appear to validate because both series have the same seasonal cycle.

What *is* transferable is the distribution. So this module does quantile
mapping, fitted per calendar month on the historical overlap and applied
unchanged to the scenarios -- which is also precisely the BCSD family the
proposal names as the traditional comparator, so building it here supplies
that baseline as well.

Two choices worth stating:

* **Quantile *delta* mapping, not plain quantile mapping.** Plain QM forces the
  scenario onto the historical observed distribution, which silently removes
  the climate change signal -- the entire point of a projection. QDM transfers
  the model's own *change* in each quantile and applies it to the observed
  distribution (Cannon et al., 2015). Where the future equals the historical
  it reduces to plain QM.
* **Per calendar month.** Central Texas rainfall is convective in summer and
  frontal in winter with different distributions, and a single annual mapping
  would correct neither well.

Dry days are handled explicitly: precipitation has an atom at zero, and
interpolating quantiles through it produces drizzle that is not there. The
wet-day threshold is applied first and the frequency of wet days is corrected
separately.

**Status: NOT VALIDATED. Do not use these fields downstream.**

Frequency is now right. Over the ACCESS-CM2 ssp245 domain the corrected
wet-day frequency is 2.0 % below observed against the model's own -0.8 %
change, after three fixes recorded in the git history: tau taken from the
future rather than the historical distribution, the wet threshold anchored on
observations and scaled by the model's own frequency change, and that
threshold floored at the wet cut so the quantile cannot land inside
precipitation's atom at zero.

**Intensity is not right, and the defect is open.** The corrected wet-day mean
comes out 22.2 % above observed. The model's own change on the days QDM
actually operates on -- its top 27.3 % -- is +6.4 %, so the correction is
inflating the signal roughly threefold rather than transferring it. Measured
for reference: the model's change is +7.1 % over all its wet days, +6.4 % over
the top 27.3 %, +7.8 % over the top 10 % and +10.1 % over the top 2 %, so the
tail does intensify faster, but nowhere near enough to explain 22 %.

The likely locus is the delta term. It is a per-value ratio multiplied by a
quantile of the observed distribution, and the two are positively correlated,
so the mean of the product exceeds the product of the means. The clip at 10.0
and the asymmetry between how the historical and future wet sets are selected
are both candidates. None of this is demonstrated.

    python -m src.projections --hist 'nexgddp/pr_ACCESS-CM2_historical_*.nc' \\
        --future 'nexgddp/pr_ACCESS-CM2_ssp245_*.nc' \\
        --obs /work/.../dsc-power-f2/aorc_aligned_1km.nc \\
        --out results/projections/ACCESS-CM2_ssp245.nc
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import xarray as xr

KGM2S_TO_MM = 86400.0
WET = 0.1            # mm/day
Q = np.concatenate([[0.0], np.linspace(0.005, 0.995, 199), [1.0]])


def load_pr(pattern: str) -> xr.DataArray:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files match {pattern}")
    ds = xr.open_mfdataset(paths, combine="by_coords")
    v = "pr" if "pr" in ds else list(ds.data_vars)[0]
    da = ds[v]
    if str(da.attrs.get("units", "")).startswith("kg"):
        da = da * KGM2S_TO_MM
        da.attrs["units"] = "mm/day"
    return da


def qdm_cell(hist: np.ndarray, fut: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Quantile delta mapping for one cell and one month.

    Returns the corrected future series. Wet-day frequency is matched first,
    then the wet-day quantiles are mapped with the model's own change carried
    through multiplicatively, which is the right form for a non-negative,
    right-skewed variable.
    """
    out = np.zeros_like(fut)
    ho, hh = obs[np.isfinite(obs)], hist[np.isfinite(hist)]
    if ho.size < 30 or hh.size < 30:
        return fut
    # 1. Wet-day frequency. The model drizzles: over central Texas ACCESS-CM2
    #    rains on 0.356 of days against an observed 0.273, with a wet-day mean
    #    24 % too low, and the two errors cancel in the annual mean. So the
    #    frequency is anchored to the observed base rate -- but scaled by the
    #    model's OWN change in wet-day frequency, which is part of the signal
    #    a projection is for.
    #
    #    Taking the threshold from the historical distribution and applying it
    #    to a shifted future does not do this: more future days clear a fixed
    #    bar simply because the distribution moved, which manufactured a 7.6 %
    #    frequency increase where the model's own change was -0.8 %.
    p_wet_obs = float(np.mean(ho >= WET))
    if p_wet_obs <= 0:
        return out
    fv = fut[np.isfinite(fut)]
    if fv.size < 30:
        return fut
    p_wet_h = float(np.mean(hh >= WET))
    p_wet_f = float(np.mean(fv >= WET))
    ratio = (p_wet_f / p_wet_h) if p_wet_h > 0 else 1.0
    p_target = float(np.clip(p_wet_obs * ratio, 0.0, 1.0))
    if p_target <= 0:
        return out
    thr = float(np.quantile(fv, 1.0 - p_target)) if p_target < 1 else WET
    # Floor at the wet cut. Precipitation has an atom at zero, and where the
    # model is wet on fewer days than the target the quantile lands inside that
    # atom -- at or below 0.1 mm. `fut >= thr` then selects dry days as well,
    # and the amount mapping turns every one of them into a wet day, because
    # the smallest observed wet quantile is well above zero. On ACCESS-CM2 over
    # this domain that happened in 7.4 % of cell-months and inflated the
    # corrected wet-day frequency by 9 %. Flooring here caps the selection at
    # the days the model itself calls wet, which is the most it can support.
    thr = max(thr, WET)
    wet_f = fut >= thr
    if not wet_f.any():
        return out
    # 2. Quantile delta on wet days.
    hw = hh[hh >= np.quantile(hh, 1.0 - p_wet_obs)] if p_wet_obs < 1 else hh
    ow = ho[ho >= WET]
    if hw.size < 20 or ow.size < 20:
        return fut
    qh = np.quantile(hw, Q)
    qo = np.quantile(ow, Q)
    fw = fut[wet_f]
    # Cannon et al. (2015): tau is where the value sits in the *future model's
    # own* distribution. Taking it from the historical distribution instead
    # makes qh_at identically fw, so delta collapses to 1 and the method
    # silently degenerates to plain quantile mapping -- which is the one thing
    # QDM exists to avoid, because it erases the change signal.
    qf = np.quantile(fw, Q)
    tau = np.interp(fw, qf, Q, left=Q[0], right=Q[-1])
    # The model's own change at that quantile, as a ratio, capped so a
    # near-zero historical quantile cannot produce an unbounded multiplier.
    qh_at = np.interp(tau, Q, qh)
    delta = np.clip(fw / np.maximum(qh_at, 1e-3), 0.1, 10.0)
    out[wet_f] = np.interp(tau, Q, qo) * delta
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hist", required=True, help="glob of model historical files")
    ap.add_argument("--future", required=True, help="glob of model scenario files")
    ap.add_argument("--obs", required=True, help="observed field on the same grid")
    ap.add_argument("--obs-var", default="precip")
    ap.add_argument("--obs-start", default="2000-01-01")
    ap.add_argument("--obs-end", default="2014-12-31")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    H = load_pr(a.hist).load()
    F = load_pr(a.future).load()
    O = xr.open_dataset(a.obs)[a.obs_var].sel(time=slice(a.obs_start, a.obs_end)).load()
    if (O.sizes["lat"], O.sizes["lon"]) != (H.sizes["lat"], H.sizes["lon"]):
        raise SystemExit(
            f"grid mismatch: obs {O.sizes['lat']}x{O.sizes['lon']} vs model "
            f"{H.sizes['lat']}x{H.sizes['lon']}. The observed field must already "
            f"be on the model grid; build it with src.multires_target.")
    print(f"historical {H.sizes['time']}d, future {F.sizes['time']}d, "
          f"obs {O.sizes['time']}d, grid {H.sizes['lat']}x{H.sizes['lon']}", flush=True)
    print(f"  raw model mean {float(H.mean()):.3f}  observed mean {float(O.mean()):.3f} mm/day",
          flush=True)

    hm = H["time.month"].values
    fm = F["time.month"].values
    om = O["time.month"].values
    Hv, Fv, Ov = H.values, F.values, O.values
    out = np.zeros_like(Fv)
    for mon in range(1, 13):
        ih, iff, io = hm == mon, fm == mon, om == mon
        if not iff.any():
            continue
        for y in range(Hv.shape[1]):
            for x in range(Hv.shape[2]):
                out[iff, y, x] = qdm_cell(Hv[ih, y, x], Fv[iff, y, x], Ov[io, y, x])
        print(f"  month {mon:2d}  corrected {int(iff.sum()):5d} days", flush=True)

    da = xr.DataArray(out, coords=F.coords, dims=F.dims, name="precip")
    da.attrs = {"units": "mm/day",
                "long_name": "quantile-delta-mapped precipitation",
                "method": ("QDM per calendar month, wet-day frequency matched "
                           "first; fitted on the historical overlap and applied "
                           "unchanged to the scenario"),
                "obs_period": f"{a.obs_start}..{a.obs_end}",
                "caveat": ("CMIP6 historical is free-running, so this is a "
                           "distributional correction, not a day-matched one")}
    print(f"  corrected future mean {float(da.mean()):.3f} mm/day "
          f"(raw future {float(F.mean()):.3f})", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset().to_netcdf(a.out)
    print("written", a.out)


if __name__ == "__main__":
    main()
