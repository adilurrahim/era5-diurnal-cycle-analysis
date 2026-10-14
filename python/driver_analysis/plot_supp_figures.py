"""
Supplementary Figures S5-S8 for Bushra et al. (#2025EA004726, Revision 2).

Reads the trend NetCDFs produced by supplemental_driver_analysis.py and makes
four figures, each with panels for January, April, July, and October
(matching the months shown in main-text Figures 1 and 2):

  S5: total cloud cover trend (Theil-Sen slope, fraction yr-1)
  S6: surface solar radiation downwards trend (J m-2 yr-1)
  S7: volumetric soil water (layer 1) trend (m3 m-3 yr-1)
  S8: low cloud cover trend (fraction yr-1)

Run on SuperMike-III after the analysis job:
    python plot_supp_figures.py
Outputs PNG files (300 dpi) in /work/nbushr1/era5_drivers/results/figures/
"""

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/work/nbushr1/era5_drivers/results"
NTGT_CSV = "/work/nbushr1/ntgt_data/processed_data/TrendAnalysisAbove_01.csv"
FIGDIR = f"{RESULTS}/figures"
MONTHS = [(1, "January"), (4, "April"), (7, "July"), (10, "October")]


def land_mask(target):
    """Boolean land mask on the driver grid, built from the NT>T land-only
    CSV grid (0.1 deg), so figures match the terrestrial analysis domain."""
    import pandas as pd
    df = pd.read_csv(NTGT_CSV)
    df.columns = [c.strip() for c in df.columns]
    grid = df.pivot_table(index="y", columns="x", values="Slope")
    da = xr.DataArray(
        grid.values.astype(np.float32),
        coords=[("latitude", grid.index.values.astype(np.float64)),
                ("longitude", grid.columns.values.astype(np.float64))])
    lon = ((da.longitude + 180) % 360) - 180
    da = da.assign_coords(longitude=lon).sortby("longitude").sortby("latitude")
    on_grid = da.interp(latitude=target.latitude, longitude=target.longitude)
    return np.isfinite(on_grid.values)

FIGS = [
    ("S5", "tcc",   "Total cloud cover trend",              "fraction yr$^{-1}$"),
    ("S6", "ssrd",  "Surface solar radiation trend",        "J m$^{-2}$ yr$^{-1}$"),
    ("S7", "swvl1", "Soil moisture (layer 1) trend",        "m$^{3}$ m$^{-3}$ yr$^{-1}$"),
    ("S8", "lcc",   "Low cloud cover trend",                "fraction yr$^{-1}$"),
]

def robust_limit(fields):
    """Symmetric color limit from the 98th percentile across all panels."""
    vals = np.concatenate([np.abs(f[np.isfinite(f)]).ravel() for f in fields])
    return np.percentile(vals, 98)

def main():
    os.makedirs(FIGDIR, exist_ok=True)
    mask = None
    for fig_id, short, title, units in FIGS:
        das, fields = [], []
        for m, _ in MONTHS:
            ds = xr.open_dataset(f"{RESULTS}/{short}_trend_{m:02d}.nc")
            da = ds["trend"]
            if mask is None:
                mask = land_mask(da)
            da = da.where(mask)
            pv = ds["spearman_p"].where(mask)
            das.append((da, pv))
            fields.append(da.values)
        vlim = robust_limit(fields)

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for ax, (m, mname), (da, pval) in zip(axes.ravel(), MONTHS, das):
            im = ax.pcolormesh(da.longitude, da.latitude, da.values,
                               cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                               shading="auto", rasterized=True)
            # stipple statistically significant areas (p < 0.05), subsampled
            sig = (pval.values < 0.05) & np.isfinite(da.values)
            yy, xx = np.where(sig)
            step = 12
            ax.plot(da.longitude.values[xx[::step]], da.latitude.values[yy[::step]],
                    ".", color="k", markersize=0.3, alpha=0.35)
            ax.set_title(f"({'abcd'[MONTHS.index((m, mname))]}) {mname}", fontsize=11)
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
            ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
            ax.tick_params(labelsize=8)
        cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cb.set_label(units, fontsize=10)
        fig.suptitle(f"Figure {fig_id}. {title}, 1981–2020 (Theil–Sen slope), "
                     f"terrestrial grid points; stippling: Spearman p < 0.05", fontsize=12)
        out = f"{FIGDIR}/Figure_{fig_id}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print("wrote", out, flush=True)
    print("ALL FIGURES DONE ->", FIGDIR)

if __name__ == "__main__":
    main()
