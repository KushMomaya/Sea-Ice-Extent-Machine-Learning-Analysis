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

def load_obs_extent(xlsx_path=OBS_EXTENT_XLSX):
    """
    Load NSIDC observational monthly sea ice workbook as a time-indexed Dataframe.
    """
    if not os.path.exists(OBS_EXTENT_XLSX):
        print(f"[!] obs extent file not found: {xlsx_path}")
        return None
    
    xls = pd.ExcelFile(xlsx_path)
    sh_sheets = [s for s in xls.sheet_names if s.endswith("-SH")]
    if not sh_sheets:
        print(f"[!] no -SH sheets found: {xls.sheet_names}")
        return None
    
    frames = []
    for sheet in sh_sheets:
        raw = pd.read_excel(xls, sheet_name=sheet, header=9)
        missing = {"year", "month", "extent", "hemisphere"} - set(raw.columns)
        if missing:
            print(f"[!] sheet '{sheet}' missing expected columns {missing} -- skipping")
            continue
        frames.append(raw[["year", "month", "extent", "hemisphere"]])
        
    if not frames:
        return None
    
    df = pd.concat(frames, ignore_index=True)
    df = df[df["hemisphere"] == S]
    df = df.dropna(subset = ["year", "month", "extent"])
    df["time"] = pd.to_datetime(dict(year = df["year"], month = df["month"], day = 1))
    df = df.sort_values("time").reset_index(drop=True)
    
    return df[["time", "extent"]]

def load_obs_concentration(obs_dir=OBS_CONC):
    """
    Load gridded NSIDC concentration files.
    """
    files = sorted(glob.glob(os.path.join(obs_dir, "*.nc")))
    if not files:
        print(f"[!] no observational concentration files found at {obs_dir}")
    return xr.open_mfdataset(files, combine="by_coords")

# ========================================================================================================================================
# Model vs Observation Bias Setup - Aligning the data from the model to the observational data for direct comparison and bias calculation
# ========================================================================================================================================

def align_model_and_obs_extent(model_extent, obs_extent_df, obs_extent_col="extent"):
    """
    Align the model derived extent time series (xarray DataArray in m^2)
    with the NSIDC observational time series (DataFrame in Mkm^2)
    so they can be compared together for the purposes of visualization and bias calculation.
    """
    model_df = model_extent.to_dataframe(name="model_extent_m2").reset_index()
    model_df["time"] = pd.to_datetime(model_df["time"]).dt.to_period("M").dt.to_timestamp() #xarray time coord -> pandas timestamp -> obs_extent time format
    
    model_df["model_extent"] = model_df["model_extent_m2"] / 1e12 #converts m^2 to Mkm^2
    
    obs_df = obs_extent_df.rename(columns={obs_extent_col: "obs_extent"}).copy()
    obs_df["time"] = pd.to_datetime(obs_df["time"]).dt.to_period("M").dt.to_timestamp()
    
    aligned = pd.merge(
        model_df[["time", "model_extent"]],
        obs_df[["time", "obs_extent"]],
        on="time",
        how="inner"
    ).sort_values("time").reset_index(drop=True)
    
    n_model, n_obs = len(model_df), len(obs_df)
    print(f"Aligned {len(aligned)} overlapping months, \n model had {n_model}, obs had {n_obs}")
    
    return aligned

def compute_extent_bias(aligned_df, model_col="model_extent", obs_col="obs_extent"):
    """
    Takes the now aligned model and observational data and calculate model extent minues observational extent.
    This creates a variable for the bias which is what the machine learning model will be trained to explain.
    Also factors for common divergence in extent/biases during specific seasons(Septemberish) by calculating
    per month bias. Negative bias = model underestimates real extent (What the CMIP model documents),
    positive bias = model overestimates real extent.
    """
    aligned_df = aligned_df.copy()
    
# ====================================================================
# Data Visualization - EDA Process to lead into feature engineering
# ====================================================================

