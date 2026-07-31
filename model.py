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

FX_VARS = ["areacello", "deptho", "sftof"]

OBS_ROOT = "observational"
OBS_EXTENT_XLSX = os.path.join(OBS_ROOT, "extent", "Sea_Ice_Index_Monthly_Data_with_Statistics_G02135_v4.0.xlsx")
OBS_CONC = os.path.join(OBS_ROOT, "concentration")

TARGET_VAR = "sic"
BIAS_VAR = "sic_bias"

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
    
def merge_model_experiment(model, experiment):
    """
    Concatenate all loaded variables corresponding to a specified model and experiment into 
    one single dataset. 
    """
    variables = MODEL_VARS.get(model)
    if not variables:
        print(f"[!] no configured variables for {model}/{experiment}")
        return None
    
    var_datasets = []
    for var in variables:
        files = file_validation(model, experiment, var)
        ds = load_variable(files)
        if ds is not None:
            var_datasets.append(ds)
    
    if not var_datasets:
        return None
    
    merged = xr.merge(var_datasets, compat="override", join="inner")
    return merged

def load_fx(model):
    """
    Loads the fixed grid metadata variables (areacello, deptho, sftof).
    Used in both piControl and historical experiments because the grid of the model 
    stays the same between experiments.
    """
    fx_datasets = []
    for fx in FX_VARS:
        pattern = os.path.join(DATA_ROOT, model, "fx", f"{fx}_*.nc")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[!] No fx file found for {model}/{fx} ({pattern})")
        else:
            fx_datasets.append(xr.open_dataset(files[0]))
    
    if not fx_datasets:
        return None
    
    return xr.merge(fx_datasets, compat="override", join="inner")

def load_all_data():
    """
    Loads all experiment/model/variables into a nested dictionary:
        data[model][experiment] -> xr.dataset
        fx[model] -> xr.dataset
        
        Returns data, fx to be used for the rest of the model building process.
    """
    data = {}
    fx = {}
    
    for model in MODELS:
        print(f"\n=== {model} ===")
        fx[model] = load_fx(model)
        
        data[model] = {}
        for experiment in EXPERIMENTS:
            print(f"Loading {model}/{experiment}: ")
            ds = merge_model_experiment(model, experiment)
            if ds is not None:
                data[model][experiment] = ds
    
    return data, fx

def load_obs_extent(xlsx_path=OBS_EXTENT_XLSX, sheet_name="Data"):
    """
    
    """
    
# ====================================================================
# Data Visualization - EDA Process to lead into feature engineering
# ====================================================================

