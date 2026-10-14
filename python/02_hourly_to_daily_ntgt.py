"""
02_hourly_to_daily_ntgt.py

Daily number of hours with temperature above the daily mean (NT>T) from
ERA5-Land hourly 2-m temperature.

Bushra, Rohli, Rahim, Mostafiz, "Rising Duration of Above-Average Temperatures
Globally: Implications for Climate-Driven Heat Exposure", Earth and Space Science.

Method (identical to the R script used for the published analysis,
original_2021_2023/Rscript_2021_hourly_to_daily_NTgtT.R)
--------------------------------------------------------------------------
For every land grid point and every UTC calendar day:
    T_mean = mean of the 24 hourly 2-m temperatures (K), rounded to 2 decimals
    Above  = number of hours with T_h >  T_mean     (this is NT>T)
    Equal  = number of hours with T_h == T_mean
    Below  = number of hours with T_h <  T_mean
Land grid points are the cells that hold data in ERA5-Land (ocean cells are
missing): 2,212,863 points on the native 0.1 x 0.1 degree grid.

In the published analysis, 39 of the 40 years were processed with the 2021
script, which rounds T_mean to 2 decimals. 1986 was re-processed in 2023 with
original_2021_2023/ncfile_process_1986_rerun.R, which does not round. Use
--round-mean none to reproduce that variant. Rounding changes Above/Below only
when an hourly value lies within 0.005 K of the daily mean.

Input
-----
One or more ERA5-Land hourly netCDF files of 2-m temperature
(01_download_era5land_t2m.py downloads one file per month). Days must lie
completely inside one file (true for monthly or yearly files).

Output (one CSV per day, same layout as the archived files)
------------------------------------------------------------
<out_dir>/<YYYY>/<YYYY-MM-DD>_average_data.csv
    x, y, average, Above, Equal, Below
x = longitude (0 to 359.9 degrees E), y = latitude. Rows follow the raster
order of the netCDF file (north to south, west to east within each latitude),
which is the row order expected by processMonthly.py and later scripts.

Usage
-----
    python 02_hourly_to_daily_ntgt.py ERA5Land_t2m_1981_01.nc [more.nc ...] --out-dir Processed_Data
    python 02_hourly_to_daily_ntgt.py "raw/*.nc" --out-dir Processed_Data --round-mean none

Requirements: Python 3.9+, numpy, pandas, xarray (with the netCDF4 or scipy
engine). Memory: about 1 GB per day processed.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

HOURS_PER_DAY = 24


def open_t2m(path):
    """Open a file and return (dataset, packed t2m variable, time coordinate name).

    The variable is read without automatic decoding so that the packed int16
    values can be decoded in float64, as R's raster package does."""
    ds = xr.open_dataset(path, mask_and_scale=False, decode_times=True)
    var = "t2m" if "t2m" in ds else "2t"
    tname = "valid_time" if "valid_time" in ds[var].dims else "time"
    lat = "latitude" if "latitude" in ds[var].dims else "lat"
    lon = "longitude" if "longitude" in ds[var].dims else "lon"
    da = ds[var].transpose(tname, lat, lon)
    # raster order: north to south; longitudes increasing
    if da[lat].values[0] < da[lat].values[-1]:
        da = da.isel({lat: slice(None, None, -1)})
    if da[lon].values[0] > da[lon].values[-1]:
        da = da.isel({lon: slice(None, None, -1)})
    return ds, da, tname, lat, lon


def decode(raw, attrs):
    """Packed integer -> float64 Kelvin, missing -> NaN."""
    out = raw.astype(np.float64)
    for key in ("_FillValue", "missing_value"):
        if key in attrs:
            out[raw == attrs[key]] = np.nan
    if "scale_factor" in attrs:
        out *= float(attrs["scale_factor"])
    if "add_offset" in attrs:
        out += float(attrs["add_offset"])
    return out


def daily_counts(temps, round_mean):
    """temps: (hours, npts) float64. Returns mean, Above, Equal, Below."""
    mean = temps.mean(axis=0)
    if round_mean is not None:
        mean = np.round(mean, round_mean)
    above = (temps > mean).sum(axis=0)
    equal = (temps == mean).sum(axis=0)
    below = (temps < mean).sum(axis=0)
    return mean, above, equal, below


def process_file(path, out_dir, round_mean, land_idx_cache, overwrite):
    ds, da, tname, lat, lon = open_t2m(path)
    attrs = {**da.attrs, **da.encoding}
    times = pd.DatetimeIndex(da[tname].values)
    days = times.normalize()

    # Land mask from the first time step: cells with data
    first = decode(da.isel({tname: 0}).values, attrs).ravel()
    land = np.flatnonzero(np.isfinite(first))
    if land_idx_cache.get("idx") is None:
        land_idx_cache["idx"] = land
        lon2d, lat2d = np.meshgrid(da[lon].values.astype(np.float64),
                                   da[lat].values.astype(np.float64))
        land_idx_cache["x"] = np.round(lon2d.ravel()[land], 4)
        land_idx_cache["y"] = np.round(lat2d.ravel()[land], 4)
        print(f"land grid points: {land.size:,}")
    elif not np.array_equal(land, land_idx_cache["idx"]):
        raise ValueError(f"{path}: land mask differs from the first file")

    for day in days.unique():
        sel = np.flatnonzero(days == day)
        fname = os.path.join(out_dir, f"{day:%Y}", f"{day:%Y-%m-%d}_average_data.csv")
        if os.path.exists(fname) and not overwrite:
            continue
        if sel.size != HOURS_PER_DAY:
            print(f"  WARNING {day:%Y-%m-%d}: {sel.size} hourly fields in {os.path.basename(path)}",
                  file=sys.stderr)
        raw = da.isel({tname: sel}).values.reshape(sel.size, -1)[:, land]
        temps = decode(raw, attrs)
        if np.isnan(temps).any():
            raise ValueError(f"{day:%Y-%m-%d}: missing values at land points")
        mean, above, equal, below = daily_counts(temps, round_mean)

        os.makedirs(os.path.dirname(fname), exist_ok=True)
        pd.DataFrame({
            "x": land_idx_cache["x"],
            "y": land_idx_cache["y"],
            "average": mean,
            "Above": above.astype(np.int8),
            "Equal": equal.astype(np.int8),
            "Below": below.astype(np.int8),
        }).to_csv(fname, index=False, float_format="%.10g")
        print(f"  {day:%Y-%m-%d}: {sel.size} h, mean NT>T = {above.mean():.3f} h")
    ds.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="ERA5-Land hourly t2m netCDF file(s) or glob pattern(s)")
    p.add_argument("--out-dir", default="Processed_Data", help="output folder (default: Processed_Data)")
    p.add_argument("--round-mean", default="2",
                   help="decimals for the daily mean before counting (default 2, as in the 2021 run); "
                        "'none' for no rounding (1986 re-run)")
    p.add_argument("--overwrite", action="store_true", help="recompute days whose CSV already exists")
    a = p.parse_args()

    round_mean = None if a.round_mean.lower() == "none" else int(a.round_mean)
    files = sorted(f for pat in a.inputs for f in (glob.glob(pat) or [pat]))
    cache = {}
    for f in files:
        print(f)
        process_file(f, a.out_dir, round_mean, cache, a.overwrite)


if __name__ == "__main__":
    main()
