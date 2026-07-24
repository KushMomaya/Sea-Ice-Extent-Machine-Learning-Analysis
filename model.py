import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import glob

DATA_ROOT = "data"

MODELS = ["GFDL-CM3", "GFDL-ESM2M", "CMCC-CESM"]
EXPERIMENTS = ["piControl", "historical"]

MODEL_VARS = {
    "GFDL-CM3":   ["sic", "sit", "grFrazil", "pr", "prsn", "snoToIce", "strairx", "strairy", "streng"],
    "GFDL-ESM2M": ["sic", "sit", "grFrazil", "pr", "prsn", "snoToIce", "strairx", "strairy", "streng"],
    "CMCC-CESM":  ["sic", "sit", "ialb", "sim", "tsice", "transix", "transiy"],
}

FX_VARS = ["areacello", "deptho", "sftof"]

OBS_ROOT = "observations/nsidc"
OBS_EXTENT_CSV = os.path.join(OBS_ROOT, "sea_ice_extent_monthly.csv")
OBS_CONC_DIR = os.path.join(OBS_ROOT, "concentration")  # gridded .nc files, if downloaded

TARGET_VAR = "sic"


# ========================================================================
# Data Loading - Retrieving the data from /data into usable datasets
# ========================================================================

def find_var_files(model, experiment, var):
    pattern = os.path.join(DATA_ROOT, model, experiment, f"{var}_*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  [!] no files found for {model}/{experiment}/{var} ({pattern})")
    return files
 
 
def load_variable(files):
    if not files:
        return None
    return xr.open_mfdataset(files, combine="by_coords", chunks={"time": 12})
 
 
def load_model_experiment(model, experiment, variables=None):
    """
    Load and merge all variables for one model+experiment into a single
    xr.Dataset aligned on the shared grid/time coords.
    """
    variables = variables or MODEL_VARS.get(model, ["sic", "sit"])
    var_datasets = []
 
    for var in variables:
        files = find_var_files(model, experiment, var)
        ds = load_variable(files)
        if ds is not None:
            var_datasets.append(ds)
 
    if not var_datasets:
        return None
 
    # merge, allowing for minor coordinate mismatches between variables
    merged = xr.merge(var_datasets, compat="override", join="inner")
    return merged
 
 
def load_fx(model):
    """
    Load fixed grid metadata (areacello, deptho, sftof) for a model.
    These have no time dimension -- needed for area-weighted extent.
    """
    fx_datasets = []
    for var in FX_VARS:
        pattern = os.path.join(DATA_ROOT, model, "fx", f"{var}_*.nc")
        files = sorted(glob.glob(pattern))
        if files:
            fx_datasets.append(xr.open_dataset(files[0]))
        else:
            print(f"  [!] no fx file found for {model}/{var} ({pattern})")
 
    if not fx_datasets:
        return None
    return xr.merge(fx_datasets, compat="override", join="inner")
 
 
def load_all_data():
    """
    Load everything into a nested dict:
        data[model][experiment] -> xr.Dataset (time-varying vars)
        fx[model]                -> xr.Dataset (grid metadata)
    """
    data = {}
    fx = {}
 
    for model in MODELS:
        print(f"\n=== {model} ===")
        fx[model] = load_fx(model)
 
        data[model] = {}
        for experiment in EXPERIMENTS:
            print(f"Loading {model}/{experiment}...")
            ds = load_model_experiment(model, experiment)
            if ds is not None:
                data[model][experiment] = ds
 
    return data, fx

# ====================================================================
# Data Visualization - EDA Process to lead into feature engineering
# ====================================================================

