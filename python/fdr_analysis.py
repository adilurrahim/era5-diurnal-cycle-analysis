"""
False discovery rate (FDR) analysis of the grid-point NT>T trend tests for
Bushra et al., "Rising Duration of Above-Average Temperatures Globally"
(Earth and Space Science, #2025EA004726RR, third revision).

Produces Table S1 (CSV) and Figure S9 (PNG) from the public Zenodo files
(Rahim, 2025a; https://doi.org/10.5281/zenodo.16888540):
    processed_data/TrendAnalysisAbove_MM.csv
    columns: x (0-360), y, Spearman rho, Spearman pval, Slope, MK pval

Method
------
Benjamini and Hochberg (1995) FDR at alpha_FDR = 0.10 (Wilks, 2016, recommends
alpha_FDR = 2 x alpha_global for spatially correlated fields), applied
separately to the Spearman p-values of each calendar month. Areas are
cosine-latitude weighted fractions of all land grid points.
Also checks that the stored p-values equal the two-sided t approximation
with n - 2 = 38 degrees of freedom computed from the stored rho (needs scipy).

Usage
-----
    python fdr_analysis.py <path to processed_data.zip or processed_data folder> <output folder>
"""
import os
import sys
import zipfile
import numpy as np
import pandas as pd

ALPHA_FDR = 0.10
N_YEARS = 40
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
AFRICA = (-35, 37, -18, 52)          # lat_min, lat_max, lon_min, lon_max (as in the driver analysis)
SIBERIA_KOREA = (35, 75, 60, 140)


def read_month(src, m):
    name = f"TrendAnalysisAbove_{m:02d}.csv"
    if src.endswith(".zip"):
        with zipfile.ZipFile(src) as z:
            with z.open(f"processed_data/{name}") as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(os.path.join(src, name))
    df.columns = [c.strip() for c in df.columns]
    return df


def bh_threshold(p, q=ALPHA_FDR):
    ps = np.sort(p)
    crit = q * np.arange(1, ps.size + 1) / ps.size
    ok = np.nonzero(ps <= crit)[0]
    return float(ps[ok[-1]]) if ok.size else 0.0


def box(lat, lon, b):
    return (lat >= b[0]) & (lat <= b[1]) & (lon >= b[2]) & (lon <= b[3])


def main(src, out):
    os.makedirs(out, exist_ok=True)
    rows, maps = [], {}
    for m in range(1, 13):
        df = read_month(src, m)
        lat = df["y"].values.astype(float)
        lon = ((df["x"].values.astype(float) + 180) % 360) - 180
        p = df["Spearman pval"].values.astype(float)
        rho = df["Spearman rho"].values.astype(float)
        s = df["Slope"].values.astype(float)
        mkp = df["MK pval"].values.astype(float)
        try:
            from scipy import stats
            t = rho * np.sqrt((N_YEARS - 2) / np.clip((1 + rho) * (1 - rho), 1e-15, None))
            pcheck = float(np.max(np.abs(2 * stats.t.sf(np.abs(t), N_YEARS - 2) - p)))
        except ImportError:
            pcheck = np.nan
        w = np.cos(np.deg2rad(lat))
        W = w.sum()
        th = bh_threshold(p)
        fdr = p <= th
        af = box(lat, lon, AFRICA)
        sk = box(lat, lon, SIBERIA_KOREA)
        fd = fdr & (s < 0)
        rows.append({
            "month": MONTHS[m - 1],
            "n_grid_points": p.size,
            "n_distinct_p": int(np.unique(p).size),
            "max_abs_diff_p_vs_t_approx": pcheck,
            "uncorr_increase_pct": 100 * w[(p < 0.05) & (s > 0)].sum() / W,
            "uncorr_decrease_pct": 100 * w[(p < 0.05) & (s < 0)].sum() / W,
            "uncorr_total_pct": 100 * w[p < 0.05].sum() / W,
            "pFDR": th,
            "fdr_increase_pct": 100 * w[fdr & (s > 0)].sum() / W,
            "fdr_decrease_pct": 100 * w[fd].sum() / W,
            "fdr_total_pct": 100 * w[fdr].sum() / W,
            "africa_share_of_fdr_area_pct": 100 * w[fdr & af].sum() / w[fdr].sum(),
            "siberia_korea_share_of_fdr_decrease_area_pct": 100 * w[fd & sk].sum() / w[fd].sum() if fd.any() else np.nan,
            "africa_land_share_pct": 100 * w[af].sum() / W,
            "siberia_korea_land_share_pct": 100 * w[sk].sum() / W,
            "spearman_mk_agreement_pct": 100 * np.mean((p < 0.05) == (mkp < 0.05)),
        })
        val = np.zeros(p.size, np.int8)
        val[fdr & (s > 0)] = 1
        val[fd] = -1
        maps[m] = (lat, lon, val)
        print(MONTHS[m - 1], "done", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "Table_S1_FDR.csv"), index=False, float_format="%.6g")
    plot(maps, os.path.join(out, "Figure_S9_FDR.png"))
    print(tab.round(3).to_string())


def plot(maps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap(["#042C53", "#D3D1C7", "#7B1818"])
    fig, axes = plt.subplots(4, 3, figsize=(7.5, 8.6), dpi=300)
    plt.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.09, wspace=0.12, hspace=0.32)
    for m in range(1, 13):
        lat, lon, val = maps[m]
        ix = np.clip(np.rint((np.round(lon, 1) + 180) / 0.1).astype(int), 0, 3599)
        iy = np.clip(np.rint((np.round(lat, 1) + 90) / 0.1).astype(int), 0, 1800)
        img = np.full((1801, 3600), np.nan, np.float32)
        img[iy, ix] = val
        ax = axes.ravel()[m - 1]
        ax.imshow(np.ma.masked_invalid(img), origin="lower", extent=[-180, 180, -90, 90], cmap=cmap,
                  vmin=-1.5, vmax=1.5, interpolation="nearest", aspect="auto")
        ax.set_title(f"({'abcdefghijkl'[m - 1]}) {MONTHS[m - 1]}", fontsize=8, fontweight="bold", loc="left", pad=2)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xticks(range(-120, 121, 60)); ax.set_yticks(range(-60, 61, 30))
        ax.set_xticklabels([f"{v}°" for v in range(-120, 121, 60)])
        ax.set_yticklabels([f"{v}°" for v in range(-60, 61, 30)])
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        ax.grid(color="#E6E6E6", linewidth=0.3); ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color("#999999")
        if (m - 1) % 3 == 0:
            ax.set_ylabel("Latitude", fontsize=6.5, labelpad=1)
        if m > 9:
            ax.set_xlabel("Longitude", fontsize=6.5, labelpad=1)
    handles = [Patch(color="#7B1818", label="Increasing NT>T, significant after FDR control"),
               Patch(color="#042C53", label="Decreasing NT>T, significant after FDR control"),
               Patch(color="#D3D1C7", label="Not significant after FDR control")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.53, 0.005))
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
