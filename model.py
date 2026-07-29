import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr # Specifally good for working with NetCDF files
import os
import glob # Pathname pattern matching

DATA_ROOT = "data"

MODELS = ["GFDL-CM3", "GFDL-ESM2M", "CMCC-CESM"]
EXPERIMENTS = ["piControl", "historical"]

MODEL_VARS = {
    "GFDL-CM3":   ["sic", "sit", "grFrazil", "pr", "prsn", "snoToIce", "strairx", "strairy", "streng"],
    "GFDL-ESM2M": ["sic", "sit", "grFrazil", "pr", "prsn", "snoToIce", "strairx", "strairy", "streng"],
    "CMCC-CESM":  ["sic", "sit", "ialb", "sim", "tsice", "transix", "transiy"],
}

FX_VARS = ["areacello", "deptho", "sftof", "gridspec"]

OBS_ROOT = "observational/nsidc"
OBS_EXTENT_CSV = os.path.join(OBS_ROOT, "Sea_Ice_Index_Monthly_Data_with_Statistics_G02135_v4.0.xlsx")
TARGET_VAR = "sic"


# ========================================================================
# Data Loading - Retrieving the data from /data into usable datasets
# ========================================================================

def file_validation(model, experiment, var):
    """
    Locate all NetCDF files for a combination of model, experiment, var. 
    CMIP5 ouput is split across many files, one per each time chunk, so a 
    single variable's full record is split across multiple files. Using glob 
    we can capture all the files of a single variable.
    """
    pattern = os.path.join(DATA_ROOT, model, experiment, f"{var}_*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[!] no files found for {model}/{experiment}/{var} ({pattern})")
    return files

def load_variable(files):
    """
    Open and concatenate all of the NetCDF files into one xarray dataset.
    Uses the by_coords parameter to force a stitch across each file's coordinate
    data instead of relying on the raw sorting by time periods. The chunks=12 parameter
    loads the data in chunks divided by year instead of loading everything into RAM immediately.
    """
    if not files:
        return None
    return xr.open_mfdataset(files, combine="by_coords", chunks={"time": 12})
    

# ====================================================================
# Data Visualization - EDA Process to lead into feature engineering
# ====================================================================

