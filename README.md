[![DOI](https://zenodo.org/badge/1039348359.svg)](https://doi.org/10.5281/zenodo.16888577)

# ERA5 Diurnal Cycle Analysis: Hours Above Daily Mean Temperature

This repository contains the processing and plotting scripts used in the study:

**Bushra, N., Rohli, R. V., Rahim, M. A., and Mostafiz, R. B. Rising Duration of Above-Average Temperatures Globally: Implications for Climate-Driven Heat Exposure. *Earth and Space Science*.**

The analysis investigates global trends, 1981–2020, in the number of hours per day in which the hourly 2-m air temperature exceeds the daily mean (NT>T), using ERA5-Land hourly data on its native 0.1° × 0.1° grid (2,212,863 land grid points).

---

## Repository Contents

- `python/` – Python scripts for processing ERA5-Land hourly data and for the statistical analysis
- `r/` – R scripts using **ggplot2** for visualization and figure creation

## Processing chain

| Step | Script | Output |
|---|---|---|
| 1 | `python/01_download_era5land_t2m.py` | monthly ERA5-Land hourly 2-m temperature (netCDF) |
| 2 | `python/02_hourly_to_daily_ntgt.py` | daily NT>T per land grid point (`YYYY-MM-DD_average_data.csv`) |
| 3 | `python/processMonthly.py` | monthly mean NT>T per grid point (`Monthly/MM/YYYY.csv`) |
| 4 | `python/trendAnalysis.py` | grid-point Spearman and Theil–Sen trends |
| 5 | `python/03_global_mean_table1.py` | cosine-latitude weighted global-mean series and Table 1 |
| 6 | `python/fdr_analysis.py` | false discovery rate control of grid-point trends (Table S1, Figure S9) |
| 7 | `python/nonlinearity_variability_analysis.py` | non-linear trends and interannual variability |
| – | `python/DurbinWatsonTest.py` | serial correlation of residuals |
| – | `r/ggplotting_trend` | maps of trends and slopes |

**NT>T definition:** for each land grid point and UTC calendar day, the daily mean is the mean of the 24 hourly 2-m temperatures; NT>T is the number of hours with temperature above that mean.

**Units:** NT>T is in hours per day. Table 1 slopes are in hours per day per decade; grid-point slope maps are in hours per day per year.

---

## Data Source

Hourly 2-m air temperature data were obtained from **ERA5-Land hourly data from 1950 to present** (Copernicus Climate Change Service), available from the Copernicus Climate Data Store: https://cds.climate.copernicus.eu.

**Citation for dataset:**
Muñoz Sabater, J. (2019). *ERA5-Land hourly data from 1950 to present* [Dataset]. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). https://doi.org/10.24381/cds.e2161bac

The processed dataset generated in this study is archived at Zenodo: https://doi.org/10.5281/zenodo.16888541

All processing scripts are openly available in the GitHub repository (https://github.com/adilurrahim/era5-diurnal-cycle-analysis) and preserved with Zenodo: https://doi.org/10.5281/zenodo.16888577

---
