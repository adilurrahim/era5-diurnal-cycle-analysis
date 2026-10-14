"""
03_global_mean_table1.py

Global-mean NT>T series and Table 1 statistics.

For each year (1981-2020) and each calendar month, and for all days of the
year ("Overall"), the global terrestrial mean NT>T is the mean over the land
grid points in each 0.1 degree latitude band, weighted by the cosine of the
band latitude (Rahim et al., 2024). The trend of each 40-value series is
described by the Theil-Sen slope (pymannkendall, equal to the median of all
pairwise slopes) and the Spearman rank correlation with year
(scipy.stats.spearmanr; p from the t distribution with n - 2 = 38 df).

Units: NT>T is a mean daily count, in hours per day. Table 1 reports the
Theil-Sen slope multiplied by 10, i.e. hours per day per decade. The
grid-point slope maps (TrendAnalysisAbove_MM.csv, Figure 1) are per year.

Input (from processMonthly.py and the annual averaging step):
    <data_dir>/Monthly/<MM>/<YYYY>.csv   columns x, y, Above, Below
    <data_dir>/Difference/<YYYY>.csv     columns x, y, Above, Below  (annual mean of daily values)
Output:
    global_mean_NTgtT_series.csv   40 years x (12 months + Overall), hours per day
    table1_reproduced.csv          Theil-Sen slope per year and per decade, Spearman r, p

Usage:
    python 03_global_mean_table1.py --data-dir Data --out-dir results
"""

import argparse
import os

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats

YEARS = list(range(1981, 2021))
MONTHS = [f"{m:02d}" for m in range(1, 13)]
NAMES = ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"]


def band_weighted_mean(path):
    df = pd.read_csv(path, usecols=["y", "Above"])
    band = df.groupby("y")["Above"].mean()
    w = np.cos(np.deg2rad(band.index.values))
    return float((band.values * w).sum() / w.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="Data")
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    series = {}
    for mm, name in zip(MONTHS, NAMES):
        series[name] = [band_weighted_mean(os.path.join(a.data_dir, "Monthly", mm, f"{y}.csv")) for y in YEARS]
        print(name, flush=True)
    series["Overall"] = [band_weighted_mean(os.path.join(a.data_dir, "Difference", f"{y}.csv")) for y in YEARS]
    out = pd.DataFrame(series, index=pd.Index(YEARS, name="Year"))
    out.to_csv(os.path.join(a.out_dir, "global_mean_NTgtT_series.csv"))

    rows = []
    for col in ["Overall"] + NAMES:
        rho, pval = stats.spearmanr(out[col].values, YEARS)
        slope = mk.original_test(out[col].values).slope
        rows.append([col, slope, 10 * slope, rho, pval])
    pd.DataFrame(rows, columns=["Period", "TheilSen_hours_per_day_per_year",
                                "TheilSen_hours_per_day_per_decade", "Spearman_r", "p_value"]
                 ).to_csv(os.path.join(a.out_dir, "table1_reproduced.csv"), index=False)


if __name__ == "__main__":
    main()
