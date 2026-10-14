"""
Non-linearity, variability and multiplicity analysis for
Bushra et al., "Rising Duration of Above-Average Temperatures Globally"
(Earth and Space Science, #2025EA004726RR, third revision)

Purpose
-------
Responds to the Editor's requests (Sara Pryor, Oct 2026):
  (1) define the Spearman test, how p-values are computed, and correct
      for multiplicity;
  (2) explore whether NT>T variability exists that is not a secular
      (linear) trend, i.e., test for non-linear trends and quantify
      interannual variability.

It also re-computes the Table 1 global-mean statistics so the units of
the Table 1 slopes can be checked (the Table 1 slopes are about ten times
the band-weighted mean of the grid-point Theil-Sen slopes archived on
Zenodo; please confirm whether Table 1 is in hours per year or per decade).

Input
-----
The per-grid-point monthly files written by processMonthly.py in the
GitHub repository adilurrahim/era5-diurnal-cycle-analysis:
    <MONTHLY_DIR>/<MM>/<YYYY>.csv    columns: x, y, Above, Below
(one row per land grid point, 2,212,863 rows; "Above" = monthly mean of
daily NT>T for that year and calendar month).

Optional: a Nino 3.4 index file (see NINO34_FILE below) to correlate the
detrended global-mean NT>T series with ENSO. If the file is absent, the
ENSO step is skipped.

Run
---
    python nonlinearity_variability_analysis.py
Memory: about 2-3 GB per month (40 x 2.2 million float32 values).
Outputs go to OUT_DIR (CSV tables, NPZ maps, PNG figures).

Methods (all two-sided, n = 40 years, 1981-2020)
-------
Global mean: band-weighted mean exactly as described in the manuscript
  (mean NT>T over land points in each 0.1 deg latitude band, weighted by
  the cosine of the band latitude). A cell-area-weighted mean is also
  reported for comparison.
Trend: Theil-Sen slope (scipy.stats.theilslopes), Spearman rho and p
  (scipy.stats.spearmanr: t-distribution, n-2 df), OLS slope.
Non-linearity: (a) quadratic term added to the linear model, F-test
  (equivalently t-test, n-3 df) and AIC comparison; (b) Pettitt (1979)
  change-point test; (c) LOWESS smooth for display.
Variability: SD of residuals about the Theil-Sen line; lag-1
  autocorrelation of OLS residuals; signal-to-noise = |slope x 39| / SD.
Multiplicity: Benjamini-Hochberg false discovery rate at
  alpha_FDR = 0.10 (Wilks 2016 recommends alpha_FDR = 2 x alpha_global
  for spatially correlated fields), applied to the grid-point p-values of
  (i) the all-days Spearman trend test (Figure 2a) and (ii) the quadratic
  term.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------
# CONFIG: edit these three paths
# ----------------------------------------------------------------------
MONTHLY_DIR = "Analysis/Data/Monthly"          # contains 01/ ... 12/
OUT_DIR = "Analysis/NonlinearityResults"
NINO34_FILE = "nina34.anom.data"               # optional, see read_nino34()

YEARS = np.arange(1981, 2021)
N = YEARS.size
MONTHS = [f"{m:02d}" for m in range(1, 13)]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
ALPHA_FDR = 0.10


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def load_month(mm):
    """Return x, y (1-D) and A (40 x npts float32) for one calendar month."""
    x = y = None
    rows = []
    for yr in YEARS:
        df = pd.read_csv(f"{MONTHLY_DIR}/{mm}/{yr}.csv")
        df.columns = [c.strip() for c in df.columns]
        if x is None:
            x = df["x"].values
            y = df["y"].values
        else:
            # the grid order must be identical in every year
            if not (np.allclose(df["x"].values, x) and np.allclose(df["y"].values, y)):
                raise ValueError(f"grid order differs in {mm}/{yr}.csv")
        rows.append(df["Above"].values.astype(np.float32))
    return x, y, np.vstack(rows)


def days_in_month(year, month):
    return pd.Period(f"{year}-{month:02d}").days_in_month


def read_nino34(path):
    """NOAA PSL format (https://psl.noaa.gov/data/correlation/nina34.anom.data):
    first line = 'startyear endyear', then rows 'year v1 ... v12', then
    footer lines; missing = -99.99. Returns dict {(year, month): value}."""
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as f:
        lines = f.read().strip().splitlines()
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 13:
            continue
        try:
            yr = int(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        for m, v in enumerate(vals, start=1):
            if v > -99:
                out[(yr, m)] = v
    return out


# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def band_weighted_mean(y, A):
    """Manuscript weighting: per 0.1 deg latitude band mean, cos(lat) weights."""
    lat = np.round(y, 1)
    bands, inv = np.unique(lat, return_inverse=True)
    w = np.cos(np.deg2rad(bands))
    out = np.empty(A.shape[0])
    cnt = np.bincount(inv)
    for i in range(A.shape[0]):
        bm = np.bincount(inv, weights=A[i].astype(np.float64)) / cnt
        out[i] = np.sum(bm * w) / np.sum(w)
    return out


def cell_weighted_mean(y, A):
    w = np.cos(np.deg2rad(y))
    return (A.astype(np.float64) @ w) / w.sum()


def pettitt(x):
    """Pettitt (1979) change-point test. Returns (index of last point
    before the change, approximate two-sided p)."""
    n = x.size
    r = stats.rankdata(x)
    U = np.array([2 * r[:k + 1].sum() - (k + 1) * (n + 1) for k in range(n - 1)])
    K = np.max(np.abs(U))
    k = int(np.argmax(np.abs(U)))
    p = 2.0 * np.exp(-6.0 * K ** 2 / (n ** 3 + n ** 2))
    return k, min(p, 1.0)


def lowess(t, x, frac=0.5):
    """Local linear smoother with tricube weights (single pass)."""
    n = t.size
    k = max(int(np.ceil(frac * n)), 3)
    out = np.empty(n)
    for i in range(n):
        d = np.abs(t - t[i])
        h = np.sort(d)[k - 1]
        w = np.clip(1 - (d / h) ** 3, 0, None) ** 3
        W = np.diag(w)
        X = np.column_stack([np.ones(n), t - t[i]])
        beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ x, rcond=None)[0]
        out[i] = beta[0]
    return out


def series_stats(x):
    t = YEARS.astype(float)
    tc = t - t.mean()
    q = tc ** 2 - np.mean(tc ** 2)          # orthogonal to 1 and tc
    ts = stats.theilslopes(x, t)
    sp = stats.spearmanr(t, x)
    # OLS linear and quadratic (orthogonal basis)
    b = np.sum(x * tc) / np.sum(tc ** 2)
    c = np.sum(x * q) / np.sum(q ** 2)
    res1 = x - x.mean() - b * tc
    res2 = res1 - c * q
    sse1, sse2 = np.sum(res1 ** 2), np.sum(res2 ** 2)
    F = (sse1 - sse2) / (sse2 / (N - 3))
    p_quad = stats.f.sf(F, 1, N - 3)
    aic1 = N * np.log(sse1 / N) + 2 * 2
    aic2 = N * np.log(sse2 / N) + 2 * 3
    k, p_pet = pettitt(x)
    r1 = np.corrcoef(res1[:-1], res1[1:])[0, 1]
    resid_ts = x - (ts.intercept + ts.slope * t)
    sd = np.std(resid_ts, ddof=1)
    return dict(
        theilsen_slope_h_per_yr=ts.slope,
        theilsen_slope_h_per_decade=10 * ts.slope,
        spearman_rho=sp.statistic if hasattr(sp, "statistic") else sp.correlation,
        spearman_p=sp.pvalue,
        ols_slope_h_per_yr=b,
        quad_coef=c,
        quad_F=F, quad_p=p_quad,
        aic_linear=aic1, aic_quadratic=aic2,
        delta_aic_quad_minus_lin=aic2 - aic1,
        pettitt_change_after_year=int(YEARS[k]), pettitt_p=p_pet,
        lag1_autocorr_resid=r1,
        n_eff=N * (1 - r1) / (1 + r1) if r1 < 1 else np.nan,
        detrended_sd_h=sd,
        signal_to_noise=abs(ts.slope * (N - 1)) / sd if sd > 0 else np.nan,
    )


def grid_quadratic_and_trend(A, chunk=200000):
    """Vectorized per-grid-point: OLS slope, quadratic-term p (n-3 df),
    residual SD about the linear fit, lag-1 autocorrelation of residuals,
    and Spearman p for the trend (t approx, n-2 df)."""
    t = YEARS.astype(np.float64)
    tc = t - t.mean()
    q = tc ** 2 - np.mean(tc ** 2)
    rt = stats.rankdata(t)
    rtc = rt - rt.mean()
    npts = A.shape[1]
    out = {k: np.empty(npts, np.float32) for k in
           ("ols_slope", "p_quad", "quad_coef", "resid_sd", "r1", "spearman_p")}
    for s in range(0, npts, chunk):
        X = A[:, s:s + chunk].astype(np.float64)
        xm = X.mean(axis=0)
        b = (tc @ X) / np.sum(tc ** 2)
        c = (q @ X) / np.sum(q ** 2)
        r1res = X - xm - np.outer(tc, b)
        r2res = r1res - np.outer(q, c)
        sse2 = np.sum(r2res ** 2, axis=0)
        se_c = np.sqrt(sse2 / (N - 3) / np.sum(q ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = np.where(se_c > 0, c / se_c, 0.0)
        out["p_quad"][s:s + chunk] = 2 * stats.t.sf(np.abs(tstat), N - 3)
        out["quad_coef"][s:s + chunk] = c
        out["ols_slope"][s:s + chunk] = b
        out["resid_sd"][s:s + chunk] = np.sqrt(np.sum(r1res ** 2, axis=0) / (N - 2))
        num = np.sum(r1res[:-1] * r1res[1:], axis=0)
        den = np.sum(r1res ** 2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["r1"][s:s + chunk] = np.where(den > 0, num / den, np.nan)
        # Spearman rho vs year (ranks; average ranks for ties)
        R = np.apply_along_axis(stats.rankdata, 0, X)
        Rc = R - R.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = (rtc @ Rc) / np.sqrt(np.sum(rtc ** 2) * np.sum(Rc ** 2, axis=0))
            tt = rho * np.sqrt((N - 2) / np.clip((1 + rho) * (1 - rho), 1e-15, None))
        p = 2 * stats.t.sf(np.abs(tt), N - 2)
        p[~np.isfinite(rho)] = 1.0   # constant series: no trend
        out["spearman_p"][s:s + chunk] = p
    return out


def bh_threshold(p, q=ALPHA_FDR):
    ps = np.sort(p[np.isfinite(p)])
    crit = q * np.arange(1, ps.size + 1) / ps.size
    ok = np.nonzero(ps <= crit)[0]
    return float(ps[ok[-1]]) if ok.size else 0.0


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nino = read_nino34(NINO34_FILE)
    gm_rows, grid_rows, series = [], [], {}
    allday_num = None
    allday_den = np.zeros(N)
    x0 = y0 = None
    for mi, mm in enumerate(MONTHS, start=1):
        print("month", mm, flush=True)
        x, y, A = load_month(mm)
        if x0 is None:
            x0, y0 = x, y
        w_days = np.array([days_in_month(int(yr), mi) for yr in YEARS], np.float64)
        if allday_num is None:
            allday_num = A.astype(np.float64) * w_days[:, None]
        else:
            allday_num += A.astype(np.float64) * w_days[:, None]
        allday_den += w_days
        series[MONTH_NAMES[mi - 1]] = A
        # global means
        g_band = band_weighted_mean(y, A)
        g_cell = cell_weighted_mean(y, A)
        st = series_stats(g_band)
        st.update(series=MONTH_NAMES[mi - 1], weighting="band (manuscript)")
        if nino is not None:
            en = np.array([nino.get((int(yr), mi), np.nan) for yr in YEARS])
            ok = np.isfinite(en)
            resid = g_band - (stats.theilslopes(g_band, YEARS).intercept
                              + stats.theilslopes(g_band, YEARS).slope * YEARS)
            r = stats.spearmanr(en[ok], resid[ok])
            st.update(enso_spearman_rho=r[0], enso_spearman_p=r[1])
        gm_rows.append(st)
        stc = series_stats(g_cell)
        stc.update(series=MONTH_NAMES[mi - 1], weighting="cell area")
        gm_rows.append(stc)
        pd.DataFrame({"year": YEARS, "global_mean_band": g_band,
                      "global_mean_cell": g_cell}).to_csv(
            f"{OUT_DIR}/global_mean_series_{mm}.csv", index=False)
        # grid level
        g = grid_quadratic_and_trend(A)
        wcell = np.cos(np.deg2rad(y))
        W = wcell.sum()
        thq = bh_threshold(g["p_quad"])
        sigq = g["p_quad"] <= thq
        grid_rows.append(dict(
            series=MONTH_NAMES[mi - 1],
            frac_area_quad_p05_uncorrected=float(wcell[g["p_quad"] < 0.05].sum() / W),
            pFDR_quad=thq,
            frac_area_quad_FDR=float(wcell[sigq].sum() / W),
            frac_area_quad_FDR_convex=float(wcell[sigq & (g["quad_coef"] > 0)].sum() / W),
            frac_area_quad_FDR_concave=float(wcell[sigq & (g["quad_coef"] < 0)].sum() / W),
            area_mean_resid_sd_h=float((wcell * g["resid_sd"]).sum() / W),
            area_mean_r1=float(np.nansum(wcell * g["r1"]) / W),
        ))
        np.savez_compressed(f"{OUT_DIR}/grid_{mm}.npz", x=x, y=y, **g)
        del A
    # all-days series (day-weighted mean of monthly means for each year)
    Aall = (allday_num / allday_den[:, None]).astype(np.float32)
    g_band = band_weighted_mean(y0, Aall)
    st = series_stats(g_band)
    st.update(series="All days", weighting="band (manuscript)")
    gm_rows.insert(0, st)
    stc = series_stats(cell_weighted_mean(y0, Aall))
    stc.update(series="All days", weighting="cell area")
    gm_rows.insert(1, stc)
    pd.DataFrame({"year": YEARS, "global_mean_band": g_band}).to_csv(
        f"{OUT_DIR}/global_mean_series_alldays.csv", index=False)
    g = grid_quadratic_and_trend(Aall)
    wcell = np.cos(np.deg2rad(y0))
    W = wcell.sum()
    th = bh_threshold(g["spearman_p"])
    thq = bh_threshold(g["p_quad"])
    sig = g["spearman_p"] <= th
    grid_rows.insert(0, dict(
        series="All days",
        frac_area_trend_p05_uncorrected=float(wcell[g["spearman_p"] < 0.05].sum() / W),
        pFDR_trend=th,
        frac_area_trend_FDR=float(wcell[sig].sum() / W),
        frac_area_trend_FDR_increasing=float(wcell[sig & (g["ols_slope"] > 0)].sum() / W),
        frac_area_trend_FDR_decreasing=float(wcell[sig & (g["ols_slope"] < 0)].sum() / W),
        frac_area_quad_p05_uncorrected=float(wcell[g["p_quad"] < 0.05].sum() / W),
        pFDR_quad=thq,
        frac_area_quad_FDR=float(wcell[g["p_quad"] <= thq].sum() / W),
        frac_area_quad_FDR_convex=float(wcell[(g["p_quad"] <= thq) & (g["quad_coef"] > 0)].sum() / W),
        frac_area_quad_FDR_concave=float(wcell[(g["p_quad"] <= thq) & (g["quad_coef"] < 0)].sum() / W),
        area_mean_resid_sd_h=float((wcell * g["resid_sd"]).sum() / W),
        area_mean_r1=float(np.nansum(wcell * g["r1"]) / W),
    ))
    np.savez_compressed(f"{OUT_DIR}/grid_alldays.npz", x=x0, y=y0, **g)
    pd.DataFrame(gm_rows).to_csv(f"{OUT_DIR}/global_mean_statistics.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(f"{OUT_DIR}/grid_level_summary.csv", index=False)
    make_figure()
    print("DONE ->", OUT_DIR, flush=True)


def make_figure():
    """Global-mean NT>T series with Theil-Sen line, quadratic fit, LOWESS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("alldays", "(a) All days"), ("01", "(b) January"), ("04", "(c) April"),
              ("07", "(d) July"), ("10", "(e) October")]
    fig, axes = plt.subplots(5, 1, figsize=(6.5, 10), sharex=True, constrained_layout=True)
    t = YEARS.astype(float)
    for ax, (key, title) in zip(axes, panels):
        df = pd.read_csv(f"{OUT_DIR}/global_mean_series_{key}.csv")
        x = df["global_mean_band"].values
        ts = stats.theilslopes(x, t)
        tc = t - t.mean(); q = tc ** 2 - np.mean(tc ** 2)
        b = np.sum(x * tc) / np.sum(tc ** 2); c = np.sum(x * q) / np.sum(q ** 2)
        ax.plot(t, x, "o-", color="0.35", ms=3, lw=0.8, label="Global mean NT>T")
        ax.plot(t, ts.intercept + ts.slope * t, color="#7B1818", lw=1.4, label="Theil-Sen trend")
        ax.plot(t, x.mean() + b * tc + c * q, color="#378ADD", lw=1.2, ls="--", label="Quadratic fit")
        ax.plot(t, lowess(t, x), color="#D85A30", lw=1.2, ls=":", label="LOWESS (frac = 0.5)")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel("NT>T (hours day$^{-1}$)")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    axes[-1].set_xlabel("Year")
    fig.savefig(f"{OUT_DIR}/Figure_S10_global_mean_series.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
