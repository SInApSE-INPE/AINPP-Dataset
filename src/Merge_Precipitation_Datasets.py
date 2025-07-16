import glob
import gzip
import datetime
import os
import multiprocessing
from string import Template

import numpy as np
import pandas as pd
import xarray as xr

# --- 1. CONFIGURATION ---
# Define area of interest and time period.
LAT_MIN, LAT_MAX = -55.0, 33.0
LON_MIN, LON_MAX = -120.0, -23.0
LAT_RES = 0.1  # Latitude resolution in degrees
LON_RES = 0.1  # Longitude resolution in degrees
NUM_LAT = int((LAT_MAX - LAT_MIN) / LAT_RES)
NUM_LON = int((LON_MAX - LON_MIN) / LON_RES)
START_DATETIME_STR = "2024-01-01 00:00"
END_DATETIME_STR = "2024-12-31 23:00"  # Example: process 6 hours
OUTPUT_PATH = "/ainpp/benchmarks/latin_america"  # Change to your desired output path
BASE_PATH_IMERG_EARLY = "path/to/imerg/early/data"  # Change to your IMERG early data path
BASE_PATH_GSMAP_NRT = "path/to/gsmap/nrt/data"  # Change to your GSMaP NRT data path
BASE_PATH_GSMAP_NOW = "path/to/gsmap/now/data"  # Change to your GSMaP NOW data path
BASE_PATH_GSMAP_MVK = "path/to/gsmap/mvk/data"  # Change to your GSMaP MVK data path
BASE_PATH_CPTEC_MERGE = "path/to/cptec/merge/data"  # Change to your CPTEC merge data path
BASE_PATH_RAIN_GAUGE = "path/to/rain/gauge/data"  # Change to your rain gauge data path
NETCDF_ATTRS = {
    "title": "Latin America AINPP Dataset – Hourly Multi-Source Precipitation (GSMaP, IMERG, MERGE & Rain Gauge)",
    "description": "Hourly gridded precipitation over Latin America obtained from four satellite products (GSMaP-NOW, GSMaP-NRT, GSMaP-MVK, IMERG-Final), one blended product (MERGE) and an interpolated rain-gauge field derived from from national and state networks (CPTEC/INPE, INMET, ANA, CEMADEN, FUNCEME, AESA, EMPARN, ITEP-LAMEPE, DHME, CMRH, SEMARH/DHN, COMET, INEMA, CEMIG-SIMGE, SEAG, SIMEPAR, CIRAM, IAC). All sources are re-projected to a common 0.10° × 0.10° grid covering 55°S – 33°N and 120°W – 23 °W. Files are delivered in Binary, NetCDF-4 and ASCII format.",
    "collection_type": "cube",
    "keywords": "precipitation, rainfall, satellite, gauge, GSMaP, IMERG, CPTEC",
    "Conventions": "CF-1.8",
    "institution": "AINPP",
    "source": "GSMaP_NOW v3, GSMaP_NRT v7, GSMaP_MVK v7, IMERG Final V07, CPTEC-MERGE 0.25°, in-situ gauge network (CPTEC/INPE; INMET; ANA; CEMADEN; FUNCEME/CE; AESA/PB; EMPARN/RN; ITEP/LAMEPE/PE; DHME/PI; CMRH/SE; SEMARH/DHN/AL; COMET/RJ; INEMA/BA; CEMIG-SIMGE/MG; SEAG/ES; SIMEPAR/PR; CIRAM/SC; IAC/SP)",
    "doi": "null",
    "license": "CC-BY-4.0",
    "product_version": "1.0",
    "n_stations": None,
    "x": 970,
    "y": 880,
    "NW_lat": 33.0,
    "NW_lon": -120.0,
    "SE_lat": -55.0,
    "NW_lat": 33.0,
    "dy": 0.1,
    "dx": 0.1,
    "subsatellite_longitude": "null",
    "orbital_slot": "null", 
    "platform_ID": "null",
    "scene_id": "Latin America",
    "time_coverage_end": "2024-01-01T00:00:00Z",
    "time_coverage_start": "2024-12-31T23:00:00Z",
    "aggregation": "60 min",
    "dataset_name": "precip_latam_ainpp_hourly",
    "sector": "null", 
    "geospatial_lat_units": "degrees_north",
    "geospatial_lon_units": "degrees_east",
    "temporal_resolution": "hourly",
    "acknowledgement": "WMO, JAXA, NASA, INPE, CEMADEN, FUNCEME, AESA, EMPARN, ITEP, DHME, CMRH, SEMARH, COMET, INEMA, CEMIG, SEAG, SIMEPAR, CIRAM, IAC",
}

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
        
        lons = df_filtered['lon'].to_numpy()
        lats = df_filtered['lat'].to_numpy()
        rain = df_filtered['r'].to_numpy()
        
        lat_idx = ((lats - lat_slice.start) / LAT_RES).astype(int)
        lon_idx = ((lons - lon_slice.start) / LON_RES).astype(int)

        mask = (
            (lat_idx >= 0) & (lat_idx < NUM_LAT) &
            (lon_idx >= 0) & (lon_idx < NUM_LON)
        )
        flat_idx = lat_idx[mask] * NUM_LON + lon_idx[mask]
        rain_vals = rain[mask]

        raingauge_sum = np.bincount(flat_idx, weights=rain_vals, minlength=NUM_LAT * NUM_LON)
        raingauge_count = np.bincount(flat_idx, minlength=NUM_LAT * NUM_LON)

        raingauge_sum = raingauge_sum.reshape(NUM_LAT, NUM_LON)
        raingauge_count = raingauge_count.reshape(NUM_LAT, NUM_LON)

        with np.errstate(divide="ignore", invalid="ignore"):
            interp_rain = raingauge_sum / raingauge_count
        interp_rain[raingauge_count == 0] = np.nan

        grid_lat = np.arange(lat_slice.start, lat_slice.stop, 0.1)
        grid_lon = np.arange(lon_slice.start, lon_slice.stop, 0.1)

        ds = xr.Dataset({"rain_gauge": (["lat", "lon"], interp_rain)}, coords={"lat": grid_lat, "lon": grid_lon})
        ds['n_stations'] = (["lat", "lon"], raingauge_count.reshape(NUM_LAT, NUM_LON))
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
    target_grid = xr.Dataset(coords={
        'lat': lat_out,
        'lon': lon_out
    })
    regridded_datasets = [ds.interp_like(target_grid, method='linear') for ds in loaded_datasets]
    merged_ds = xr.merge(regridded_datasets, compat='override')

    # Remove unnecessary variables before saving
    vars_to_drop = [var for var in ['hourlyPrecipRateGC', 'snowProbability', 'observationTimeFlag', 
                                   'reliabilityFlag', 'prmsl', 'surface', 'step', 
                                   'valid_time', 'Time'] if var in merged_ds]
    if vars_to_drop:
        merged_ds = merged_ds.drop_vars(vars_to_drop)
    merged_ds = merged_ds.expand_dims('time')
    attrs = NETCDF_ATTRS.copy()
    merged_ds.attrs.update(attrs)
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