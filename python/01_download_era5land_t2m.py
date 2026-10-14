"""
01_download_era5land_t2m.py

Download ERA5-Land hourly 2-m temperature (native 0.1 x 0.1 degree, global
land) from the Copernicus Climate Data Store, one netCDF file per month,
1981-2020.

Dataset: "ERA5-Land hourly data from 1950 to present",
         doi:10.24381/cds.e2161bac (Munoz-Sabater et al., 2021)
CDS name: reanalysis-era5-land, variable 2m_temperature, all 24 hours (UTC).

The data for the published analysis were downloaded from the CDS in 2021
(1986 again in January 2023) as monthly netCDF files with this content:
t2m (time, latitude, longitude), 3600 x 1801 grid, longitude 0 to 359.9,
latitude 90 to -90, ocean cells missing, 2,212,863 land cells with data.
This script makes the same request with the current CDS API.

Setup: pip install "cdsapi>=0.7" and put your CDS key in ~/.cdsapirc
(https://cds.climate.copernicus.eu/how-to-api). Each monthly file is about
9 GB; the full period is about 4.3 TB.

Usage:
    python 01_download_era5land_t2m.py --start 1981 --end 2020 --out-dir raw
"""

import argparse
import calendar
import os

import cdsapi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=1981)
    p.add_argument("--end", type=int, default=2020)
    p.add_argument("--out-dir", default="raw")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    client = cdsapi.Client()
    for year in range(a.start, a.end + 1):
        for month in range(1, 13):
            target = os.path.join(a.out_dir, f"ERA5Land_t2m_{year}_{month:02d}.nc")
            if os.path.exists(target):
                continue
            ndays = calendar.monthrange(year, month)[1]
            client.retrieve(
                "reanalysis-era5-land",
                {
                    "variable": ["2m_temperature"],
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": [f"{d:02d}" for d in range(1, ndays + 1)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                target,
            )
            print("saved", target)


if __name__ == "__main__":
    main()
