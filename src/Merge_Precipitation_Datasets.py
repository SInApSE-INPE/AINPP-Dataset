import glob
import gzip
import datetime
import os
import multiprocessing
from string import Template

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

# --- 1. CONFIGURATION ---
# Define area of interest and time period.
LAT_MIN, LAT_MAX = -55.0, 13.0
LON_MIN, LON_MAX = -83.0, -33.0
START_DATETIME_STR = "2018-01-01 00:00"
END_DATETIME_STR = "2024-12-31-23:00"  # Example: process 6 hours
OUTPUT_PATH = "path/to/output/directory"  # Change to your desired output path
BASE_PATH_IMERG_EARLY = "path/to/imerg/early/data"  # Change to your IMERG early data path
BASE_PATH_GSMAP_NRT = "path/to/gsmap/nrt/data"  # Change to your GSMaP NRT data path
BASE_PATH_GSMAP_NOW = "path/to/gsmap/now/data"  # Change to your GSMaP NOW data path
BASE_PATH_GSMAP_MVK = "path/to/gsmap/mvk/data"  # Change to your GSMaP MVK data path
BASE_PATH_CPTEC_MERGE = "path/to/cptec/merge/data"  # Change to your CPTEC merge data path
BASE_PATH_RAIN_GAUGE = "path/to/rain/gauge/data"  # Change to your rain gauge data path

# For optimization, set the number of parallel processes.
# Using 'None' will use all available CPU cores.
NUM_WORKERS = None

# Mapping of products to their respective path templates.
PATH_TEMPLATES = {
    "imerg_early": Template(f"{BASE_PATH_IMERG_EARLY}/$year/$month/$day/3B-HHR.MS.MRG.3IMERG.$year$month$day-*.$hour$minute.V07B.HDF5.nc4"),
    "gsmap_nrt": Template(f"{BASE_PATH_GSMAP_NRT}/$year/$month/$day/gsmap_nrt.$year$month$day.$hour$minute.dat.gz"),
    "gsmap_now": Template(f"{BASE_PATH_GSMAP_NOW}/$year/$month/$day/gsmap_now.$year$month$day.$hour$minute.dat.gz"),
    "gsmap_mvk": Template(f"{BASE_PATH_GSMAP_MVK}/$year/$month/$day/gsmap_mvk.$year$month$day.$hour$minute.v8.0000.0.nc"),
    "cptec_merge": Template(f"{BASE_PATH_CPTEC_MERGE}/MERGE_CPTEC_$year$month$day$hour.grib2"),
    "rain_gauge": Template(f"{BASE_PATH_RAIN_GAUGE}/$year/prec_obs_cptec_$year$month.txt"),
}

def find_file(template: Template, params: dict) -> str | None:
    """Find a file based on a template and date/time parameters."""
    glob_path = template.substitute(**params)
    found_files = glob.glob(glob_path)
    if not found_files:
        # Silent to avoid polluting the log in parallel mode
        return None
    return found_files[0]

# --- 2. PRODUCT-SPECIFIC PROCESSING FUNCTIONS ---

def process_imerg(filepath: str, lat_slice: slice, lon_slice: slice) -> xr.Dataset | None:
    """Load, process, and subset IMERG data."""
    try:
        ds = xr.open_dataset(filepath)
        ds = ds.sel(lat=lat_slice, lon=lon_slice)
        ds = ds.transpose('time', 'lat', 'lon').isel(time=0)
        ds = ds.rename({'precipitation': 'imerg_early'})
        return ds
    except Exception as e:
        print(f"Error processing IMERG file {filepath}: {e}")
        return None


def process_gsmap_binary(filepath: str, var_name: str, lat_slice: slice, lon_slice: slice) -> xr.Dataset | None:
    """Load and process binary GSMaP data (NRT and NOW)."""
    try:
        with gzip.open(filepath, mode='rb') as handle:
            data = np.frombuffer(handle.read(), dtype=np.float32).reshape(1200, 3600)
            data = np.roll(data, shift=1800, axis=1)[::-1]
        lats = np.linspace(-60, 60, 1200)
        lons = np.linspace(-180, 180, 3600)
        ds = xr.Dataset({var_name: (["lat", "lon"], data)}, coords={"lat": lats, "lon": lons})
        ds = ds.sel(lat=lat_slice, lon=lon_slice)
        return ds
    except Exception as e:
        print(f"Error processing binary GSMaP file {filepath}: {e}")
        return None


def process_gsmap_mvk(filepath: str, lat_slice: slice, lon_slice: slice) -> xr.Dataset | None:
    """Load, process, and subset GSMaP-MVK (NetCDF) data."""
    try:
        ds = xr.open_dataset(filepath, decode_times=False)
        ds = ds.rename({'hourlyPrecipRate': 'gsmap_mvk', 'Latitude': 'lat', 'Longitude': 'lon'})
        ds = ds.sel(lat=lat_slice, lon=lon_slice)
        ds = ds.isel(Time=0).transpose('lat', 'lon')
        return ds
    except Exception as e:
        print(f"Error processing GSMaP-MVK file {filepath}: {e}")
        return None


def process_cptec_merge(filepath: str, lat_slice: slice, lon_slice: slice) -> xr.Dataset | None:
    """Load, process, and subset MERGE-CPTEC (GRIB2) data."""
    try:
        ds = xr.open_dataset(filepath, engine='cfgrib', decode_times=False)
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon', 'rdp': 'cptec_merge'})
        ds['lon'] = ((ds['lon'] + 180) % 360) - 180
        ds = ds.sortby('lat').sortby('lon')
        ds = ds.sel(lat=lat_slice, lon=lon_slice)
        return ds
    except Exception as e:
        print(f"Error processing MERGE-CPTEC file {filepath}: {e}")
        return None


def process_rain_gauge(filepath: str, target_time: datetime.datetime, lat_slice: slice, lon_slice: slice) -> xr.Dataset | None:
    """Read rain gauge data, filter by time, and interpolate to a regular grid."""
    try:
        df = pd.read_csv(filepath, sep="\s+", skiprows=2, na_values=[-999, -999.0])
        df = df.iloc[:-3, :].dropna(subset=['lat', 'lon', 'r'])
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H')
        time_end = target_time + datetime.timedelta(hours=1)
        df_filtered = df[(df['date'] >= target_time) & (df['date'] < time_end)]

        if df_filtered.empty:
            return None
        points = df_filtered[['lon', 'lat']].to_numpy()
        values = df_filtered['r'].to_numpy()
        grid_lat = np.arange(lat_slice.start, lat_slice.stop, 0.1)
        grid_lon = np.arange(lon_slice.start, lon_slice.stop, 0.1)
        lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)
        interp_rain = griddata(points, values, (lon_mesh, lat_mesh), method="nearest", fill_value=np.nan)
        ds = xr.Dataset({"rain_gauge": (["lat", "lon"], interp_rain)}, coords={"lat": grid_lat, "lon": grid_lon})
        return ds
    except Exception as e:
        print(f"Error processing rain gauge data {filepath}: {e}")
        return None

# --- 3. ORCHESTRATION AND EXECUTION ---

def process_single_timestamp(target_time: datetime.datetime):
    """Orchestrate loading, processing, and merging of products for a single timestamp."""
    print(f"Starting processing for: {target_time.strftime('%Y-%m-%d %H:%M')}")
    datetime_params = {
        'year': target_time.strftime("%Y"),
        'month': target_time.strftime("%m"),
        'day': target_time.strftime("%d"),
        'hour': target_time.strftime("%H"),
        'minute': target_time.strftime("%M"),
    }
    lat_slice = slice(LAT_MIN, LAT_MAX)
    lon_slice = slice(LON_MIN, LON_MAX)

    PROCESSORS = {
        "imerg_early": (process_imerg, {}),
        "gsmap_nrt": (process_gsmap_binary, {"var_name": "gsmap_nrt"}),
        "gsmap_now": (process_gsmap_binary, {"var_name": "gsmap_now"}),
        "gsmap_mvk": (process_gsmap_mvk, {}),
        "cptec_merge": (process_cptec_merge, {}),
        "rain_gauge": (process_rain_gauge, {"target_time": target_time}),
    }

    loaded_datasets = []
    for product, (processor_func, kwargs) in PROCESSORS.items():
        filepath = find_file(PATH_TEMPLATES[product], datetime_params)
        if filepath:
            ds = processor_func(filepath, lat_slice=lat_slice, lon_slice=lon_slice, **kwargs)
            if ds:
                loaded_datasets.append(ds)

    if not loaded_datasets:
        print(f"No data loaded for: {target_time}. Skipping.")
        return

    lat_out = np.arange(LAT_MIN, LAT_MAX, 0.1)
    lon_out = np.arange(LON_MIN, LON_MAX, 0.1)
    target_grid = xr.Dataset(coords={"lat": lat_out, "lon": lon_out})
    regridded_datasets = [ds.interp_like(target_grid, method='linear') for ds in loaded_datasets]
    merged_ds = xr.merge(regridded_datasets, compat='override')

    # Remove unnecessary variables before saving
    vars_to_drop = [var for var in ['hourlyPrecipRateGC', 'snowProbability', 'observationTimeFlag', 
                                   'reliabilityFlag', 'prmsl', 'surface', 'step', 
                                   'valid_time', 'Time'] if var in merged_ds]
    if vars_to_drop:
        merged_ds = merged_ds.drop_vars(vars_to_drop)

    output_path = f"{OUTPUT_PATH}/{target_time.strftime('%Y')}/{target_time.strftime('%m')}/{target_time.strftime('%d')}"
    os.makedirs(output_path, exist_ok=True)
    output_filename = f"{output_path}/ainpp_south_america_{target_time.strftime('%Y%m%d%H')}.v01.nc"
    encoding = {var: {'zlib': True, 'complevel': 5} for var in merged_ds.data_vars}

    merged_ds.to_netcdf(output_filename, mode='w', format='NETCDF4', encoding=encoding)
    print(f"--> Final file saved for {target_time}: {output_filename}")


def main():
    """Main function that sets up the time range and distributes processing tasks in parallel."""
    try:
        start_time = datetime.datetime.strptime(START_DATETIME_STR, "%Y-%m-%d %H:%M")
        end_time = datetime.datetime.strptime(END_DATETIME_STR, "%Y-%m-%d-%H:%M")
    except ValueError:
        print("Error: Invalid date format. Use 'YYYY-MM-DD HH:MM' for start and 'YYYY-MM-DD-HH:MM' for end.")
        return

    # Generate a list of all hours to process
    timestamps_to_process = pd.date_range(start=start_time, end=end_time, freq='H')

    if timestamps_to_process.empty:
        print("No timestamps to process for the given range.")
        return
        
    print(f"Processing {len(timestamps_to_process)} timestamps from {start_time} to {end_time}")
    
    # Use a pool of processes to run tasks in parallel
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        pool.map(process_single_timestamp, timestamps_to_process)
        
    print("\nProcessing completed.")

if __name__ == "__main__":
    # Ensure safe multiprocessing execution
    main()