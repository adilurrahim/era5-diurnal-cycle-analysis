"""
Supplemental driver analysis for:
Bushra et al., "Rising Duration of Above-Average Temperatures Globally"
(Earth and Space Science, #2025EA004726, Revision 2)

Version 2 — adapted to the Zenodo NT>T trend format (12 monthly CSVs:
TrendAnalysisAbove_MM.csv with columns x, y, Spearman rho, Spearman pval,
Slope, MK pval; x in 0-360 longitude, 0.1 deg grid) and pre-filled for
/work/nbushr1 on SuperMike-III.

What it does
------------
1. Downloads ERA5 monthly means BY HOUR OF DAY (1981-2020, 0.25 deg) for
   cloud, radiation, and soil moisture variables via the CDS API.
2. For each variable and each calendar month: builds the 40-year annual
   series, computes the Theil-Sen slope and Spearman trend p per grid
   point (same methodology as the manuscript's NT>T trends).
3. Computes area-weighted spatial (pattern) correlations between each
   driver trend field and the corresponding monthly NT>T trend field,
   globally and for the regions discussed in the paper.
4. Writes: per-variable/month trend NetCDFs, pattern_correlations.csv,
   and quick-look outputs for supplementary Figures S5-S8.

Run order (on SuperMike-III):
  stage 1 (login node, overnight):  nohup python supplemental_driver_analysis.py download > download.log 2>&1 &
  stage 2 (batch job):              python supplemental_driver_analysis.py analyze
"""

import os
import sys
import glob
import numpy as np

# ----------------------------------------------------------------------
# CONFIG (pre-filled for nbushr1 on SuperMike-III)
# ----------------------------------------------------------------------
DATA_DIR = "/work/nbushr1/era5_drivers"
NTGT_CSV_DIR = "/work/nbushr1/ntgt_data/processed_data"   # TrendAnalysisAbove_MM.csv
OUT_DIR = "/work/nbushr1/era5_drivers/results"

YEARS = list(range(1981, 2021))
MONTHS = list(range(1, 13))

VARIABLES = {
    "tcc":   "total_cloud_cover",
    "lcc":   "low_cloud_cover",
    "hcc":   "high_cloud_cover",
    "ssrd":  "surface_solar_radiation_downwards",
    "swvl1": "volumetric_soil_water_layer_1",
}

# Regions discussed in the manuscript (lat_min, lat_max, lon_min, lon_max;
# longitudes in -180..180)
REGIONS = {
    "global": None,
    "africa": (-35, 37, -18, 52),
    "global_south": (-60, 0, -180, 180),
    "siberia_korea": (35, 75, 60, 140),
    "n_rocky_mountains": (40, 60, -125, -105),
    "antarctica": (-90, -60, -180, 180),
}

# ----------------------------------------------------------------------
# STAGE 1: DOWNLOAD (network-bound; run on login node with nohup)
# ----------------------------------------------------------------------
def download_all():
    import cdsapi
    c = cdsapi.Client()
    os.makedirs(DATA_DIR, exist_ok=True)
    for short, cds_name in VARIABLES.items():
        for yr in YEARS:
            out = f"{DATA_DIR}/{short}_{yr}.nc"
            if os.path.exists(out) and os.path.getsize(out) > 1e6:
                continue
            c.retrieve(
                "reanalysis-era5-single-levels-monthly-means",
                {
                    "product_type": "monthly_averaged_reanalysis_by_hour_of_day",
                    "variable": cds_name,
                    "year": str(yr),
                    "month": [f"{m:02d}" for m in MONTHS],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "data_format": "netcdf",
                },
                out,
            )
            print("downloaded", out, flush=True)
    print("DOWNLOAD COMPLETE", flush=True)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def to_pm180(da):
    """Convert longitude coord to -180..180 and sort lat ascending."""
    import xarray as xr
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon = ((da[lon_name] + 180) % 360) - 180
    da = da.assign_coords({lon_name: lon}).sortby(lon_name).sortby(lat_name)
    return da.rename({lon_name: "longitude", lat_name: "latitude"})

def open_year(short, yr):
    """Open one year file, return DataArray with a usable time coordinate."""
    import xarray as xr
    ds = xr.open_dataset(f"{DATA_DIR}/{short}_{yr}.nc")
    var = [v for v in ds.data_vars if ds[v].ndim >= 3][0]
    da = ds[var]
    for cand in ("time", "valid_time", "date", "forecast_reference_time"):
        if cand in da.dims:
            da = da.rename({cand: "time"})
            break
    return da

def monthly_annual_series(short, month):
    """(40, lat, lon): for each year, mean over the 24 hour-of-day fields
    of the requested calendar month."""
    import xarray as xr
    out = []
    for yr in YEARS:
        da = open_year(short, yr)
        sel = da.sel(time=da["time"].dt.month == month)
        out.append(sel.mean(dim="time"))
    ann = xr.concat(out, dim="year").assign_coords(year=YEARS)
    return to_pm180(ann)

def theilsen_field(annual, chunk=90):
    """Vectorized Theil-Sen slope (median of pairwise slopes), lat-chunked."""
    n, ny, nx = annual.shape
    ii, jj = np.triu_indices(n, k=1)
    den = (jj - ii).astype(np.float32)[:, None, None]
    out = np.empty((ny, nx), np.float32)
    for y0 in range(0, ny, chunk):
        a = annual[:, y0:y0 + chunk, :]
        out[y0:y0 + chunk] = np.nanmedian((a[jj] - a[ii]) / den, axis=0)
    return out

def spearman_p_field(annual):
    """Vectorized Spearman rho and two-sided p (t approximation, n=40)."""
    from scipy import stats
    n = annual.shape[0]
    ranks = np.apply_along_axis(stats.rankdata, 0, annual)
    tr = stats.rankdata(np.arange(n)).astype(np.float32)
    rx = ranks - ranks.mean(axis=0)
    ry = (tr - tr.mean())[:, None, None]
    rho = (rx * ry).sum(axis=0) / np.sqrt((rx ** 2).sum(axis=0) * (ry ** 2).sum())
    t = rho * np.sqrt((n - 2) / np.clip(1 - rho ** 2, 1e-12, None))
    p = 2 * stats.t.sf(np.abs(t), df=n - 2)
    return rho.astype(np.float32), p.astype(np.float32)

def load_ntgt_month(month):
    """NT>T Theil-Sen slope field for one month from the Zenodo CSVs."""
    import pandas as pd
    import xarray as xr
    f = f"{NTGT_CSV_DIR}/TrendAnalysisAbove_{month:02d}.csv"
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    grid = df.pivot_table(index="y", columns="x", values="Slope")
    da = xr.DataArray(
        grid.values.astype(np.float32),
        coords=[("latitude", grid.index.values.astype(np.float64)),
                ("longitude", grid.columns.values.astype(np.float64))],
        name="ntgt_slope",
    )
    return to_pm180(da)

def region_mask(da, box):
    if box is None:
        return da
    lat0, lat1, lon0, lon1 = box
    return da.where(
        (da.latitude >= lat0) & (da.latitude <= lat1)
        & (da.longitude >= lon0) & (da.longitude <= lon1)
    )

def pattern_corr(a, b):
    """Area-weighted spatial correlation of two DataArrays on one grid."""
    w2d = np.cos(np.deg2rad(a.latitude.values))[:, None] * np.ones(a.shape)
    av, bv = a.values, b.values
    m = np.isfinite(av) & np.isfinite(bv)
    if m.sum() < 100:
        return np.nan
    av, bv, wv = av[m], bv[m], w2d[m]
    am = np.average(av, weights=wv); bm = np.average(bv, weights=wv)
    cov = np.average((av - am) * (bv - bm), weights=wv)
    sa = np.average((av - am) ** 2, weights=wv)
    sb = np.average((bv - bm) ** 2, weights=wv)
    return float(cov / np.sqrt(sa * sb))

# ----------------------------------------------------------------------
# STAGE 2: ANALYSIS (submit as batch job)
# ----------------------------------------------------------------------
def run_analysis():
    import xarray as xr
    import csv
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for short in VARIABLES:
        for month in MONTHS:
            print(f"processing {short} month {month:02d}", flush=True)
            ann = monthly_annual_series(short, month)
            trend = theilsen_field(ann.values.astype(np.float32))
            rho, p = spearman_p_field(ann.values.astype(np.float32))
            tr = xr.DataArray(trend,
                              coords=[ann.latitude, ann.longitude],
                              dims=["latitude", "longitude"])
            xr.Dataset({"trend": tr,
                        "spearman_rho": (("latitude", "longitude"), rho),
                        "spearman_p": (("latitude", "longitude"), p)}
                       ).to_netcdf(f"{OUT_DIR}/{short}_trend_{month:02d}.nc")
            # NT>T monthly slope field, interpolated 0.1 -> 0.25 grid
            nt = load_ntgt_month(month).interp(latitude=tr.latitude,
                                               longitude=tr.longitude)
            for reg, box in REGIONS.items():
                r = pattern_corr(region_mask(tr, box), region_mask(nt, box))
                rows.append([short, month, reg, round(r, 4) if np.isfinite(r) else ""])
    with open(f"{OUT_DIR}/pattern_correlations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variable", "month", "region",
                    "pattern_correlation_vs_NTgtT_trend"])
        w.writerows(rows)
    print("ANALYSIS COMPLETE ->", f"{OUT_DIR}/pattern_correlations.csv", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "download":
        download_all()
    elif mode == "analyze":
        run_analysis()
    else:
        print("usage: python supplemental_driver_analysis.py [download|analyze]")
