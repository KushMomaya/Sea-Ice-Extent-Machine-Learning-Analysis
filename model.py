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
    "GFDL-CM3":   ["sic", "sit", "grFrazil", "strairx", "strairy", "streng"],
    "GFDL-ESM2M": ["sic", "sit", "grFrazil", "strairx", "strairy", "streng"],
    "CMCC-CESM":  ["sic", "sit", "ialb", "sim", "tsice", "transix", "transiy"],
}

FX_VARS = ["areacello", "deptho", "sftof"]

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

def plot_spatial_snapshot(da, time_index=0, title=None):
    """Single map of a variable at one time step -- sanity-checks the grid."""
    fig, ax = plt.subplots(figsize=(7, 5))
    da.isel(time=time_index).plot(ax=ax)
    ax.set_title(title or f"{da.name} at t={time_index}")
    plt.tight_layout()
    plt.show()
 
 
def plot_time_series(da, reduce_dims=("y", "x"), title=None, label=None):
    """Spatially-averaged time series -- spot trends, gaps, spikes."""
    present_dims = [d for d in reduce_dims if d in da.dims]
    series = da.mean(dim=present_dims)
    series.plot(label=label)
    plt.title(title or f"{da.name} spatial mean over time")
    plt.xlabel("time")
    plt.ylabel(da.name)
    if label:
        plt.legend()
 
 
def plot_seasonal_cycle(da, reduce_dims=("y", "x"), title=None, label=None):
    """Climatological seasonal cycle -- sea ice is strongly seasonal."""
    present_dims = [d for d in reduce_dims if d in da.dims]
    spatial_mean = da.mean(dim=present_dims)
    monthly = spatial_mean.groupby("time.month").mean("time")
    monthly.plot(label=label, marker="o")
    plt.title(title or f"{da.name} seasonal cycle")
    plt.xlabel("month")
    plt.ylabel(da.name)
    if label:
        plt.legend()
 
 
def plot_multi_model_comparison(data, var, plot_fn=plot_time_series, experiment="piControl"):
    """
    Overlay the same plot (time series or seasonal cycle) across all
    loaded models, for a shared variable, so you can see where models
    agree/disagree.
    """
    plt.figure(figsize=(9, 5))
    for model in MODELS:
        ds = data.get(model, {}).get(experiment)
        if ds is None or var not in ds:
            continue
        plot_fn(ds[var], label=model, title=f"{var} ({experiment}) across models")
    plt.tight_layout()
    plt.show()
 
 
def run_exploratory_plots(data, fx=None):
    """Standard EDA plot set across all loaded models."""
    for model in MODELS:
        ds = data.get(model, {}).get("piControl")
        if ds is None:
            continue
        if TARGET_VAR in ds:
            plot_spatial_snapshot(ds[TARGET_VAR], title=f"{model} {TARGET_VAR} snapshot")
 
    plot_multi_model_comparison(data, TARGET_VAR, plot_fn=plot_time_series)
    plot_multi_model_comparison(data, TARGET_VAR, plot_fn=plot_seasonal_cycle)