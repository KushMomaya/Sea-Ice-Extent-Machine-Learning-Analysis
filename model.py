import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr # Specifally good for working with NetCDF files
import os
import glob # Pathname pattern matching

DATA_ROOT = "data"

MODELS = ["GFDL_CM3", "GFDL_ESM2M", "CMCC_CESM"]
EXPERIMENTS = ["piControl", "historical"]

MODEL_VARS = {
    "GFDL_CM3":   ["sic", "sit", "grFrazil", "pr", "prsn", "snoToIce", "strairx", "strairy", "streng"],
    "GFDL_ESM2M": ["sic", "sit", "grFrazil", "pr", "prsn", "snoToIce", "strairx", "strairy", "streng"],
    "CMCC_CESM":  ["sic", "sit", "ialb", "sim", "tsice", "transix", "transiy"],
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
    variables = MODEL_VARS.get(model, [])
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
        pattern = os.path.join(DATA_ROOT, model, "piControl", "fx", f"{fx}_*.nc")
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
    df = df[df["hemisphere"] == "S"]
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
    aligned_df[BIAS_VAR] = aligned_df[model_col] - aligned_df[obs_col]
    aligned_df[f"{BIAS_VAR}_pct"] = 100 * aligned_df[BIAS_VAR] / aligned_df[obs_col]
    
    monthly_bias = (
        aligned_df.assign(month=pd.to_datetime(aligned_df["time"]).dt.month)
        .groupby("month")[BIAS_VAR]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean_bias", "std": "std_bias"})
    )
    print("\n Mean Bias by calendar month (model - obs, Mkm^2)")
    print(monthly_bias)
    
    return aligned_df

def plot_extents(aligned_df, model_col="model_extent", obs_col="obs_extent"):
    """
    Diagnostic model that plots the model vs observed extent time series.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10,7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    
    axes[0].plot(aligned_df["time"], aligned_df[obs_col], label="NSIDC (observed)", color="black", lw=1.5)
    axes[0].plot(aligned_df["time"], aligned_df[model_col], label="model", color="red", alpha=0.8)
    axes[0].set_ylabel("extent(Mkm^2)")
    axes[0].set_title("Model vs. Observed Sea Ice Extent")
    axes[0].legend()
    
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].plot(aligned_df["time"], aligned_df[BIAS_VAR], color="red")
    axes[1].fill_between(aligned_df["time"], aligned_df[BIAS_VAR], 0, alpha=0.2, color="red")
    axes[1].set_ylabel("bias\n(model - obs, Mkm^2)")
    axes[1].set_xlabel("time")
    
    plt.tight_layout()
    plt.show()

# ====================================================================
# Data Visualization - EDA Process to lead into feature engineering
# ====================================================================

def plot_spatial_snapshot(da, time_index=0, title=None):
    """
    Map of a single variable at one timestep
    """
    fig, ax = plt.subplots(figsize=(7,5))
    da.isel(time=time_index).plot(ax=ax)
    ax.set_title(title or f"{da.name} at t={time_index}")
    
    plt.tight_layout()
    plt.show()
    
def plot_time_series(da, reduce_dims=("y", "x"), label=None, title=None):
    """
    Spatially averaged time series to find gaps or trends in the data
    """
    present_dims = [d for d in reduce_dims if d in da.dims]
    series = da.mean(dim=present_dims)
    series.plot(label=label)
    plt.title(title or f"{da.name} spatial mean over time")
    plt.xlabel("time")
    plt.ylabel(da.name)
    if label:
        plt.legend()

def plot_seasonal_cycle(da, reduce_dims=("y", "x"), label=None, title=None):
    """
    Plots the season cycle for sea ice that should show a pattern of peaking in winter
    and bottoming in summer. 
    """
    present_dims = [d for d in reduce_dims if d in da.dims]
    spatial_mean = da.mean(dim=present_dims)
    monthly = spatial_mean.groupby("time.month").mean("time")
    monthly.plot(label=label)
    plt.title(title or f"{da.name} seasonal cycle")
    plt.xlabel("month")
    plt.ylabel(da.name)
    if label:
        plt.legend()

def plot_multi_modal_comparison(data, var, plot_fn=plot_time_series, experiment="piControl"):
    """
    Overlay the same plot across all loaded models for one variable to view
    discrepancies between models
    """
    plt.figure(figsize=(9,5))
    for model in MODELS:
        ds = data.get(model, {}).get(experiment)
        if ds is None or var not in ds:
            continue
        plot_fn(ds[var], label=model, title=f"{var} {experiment} across models")
    plt.tight_layout()
    plt.show()

def run_eda_plots(data, fx=None):
    """
    Run EDA plots across all models
    """
    for model in MODELS:
        ds = data.get(model, {}).get("piControl")
        if ds is None:
            continue
        if TARGET_VAR in ds:
            plot_spatial_snapshot(ds[TARGET_VAR], title=f"{model} {TARGET_VAR} snapshot")
    
    plot_multi_modal_comparison(data, TARGET_VAR, plot_fn=plot_time_series)
    plot_multi_modal_comparison(data, TARGET_VAR, plot_fn=plot_seasonal_cycle)


data, fx = load_all_data()
run_eda_plots(data, fx)

# ====================================================================
# EDA and Summary Statistics
# ====================================================================

def summarize_dataset(ds, model_name=""):
    """
    Print the mean, std, min, max, and NaN% per variable.
    Notably, the NaN% is not simply missing data,
    but represented areas where sea ice is physically undefined
    like land areas.
    """
    print(f"{model_name} summary")
    for var in ds.data_vars:
        da = ds[var]
        vals = da.values
        n_nan = np.isnan(vals).sum()
        n_total = vals.size
        print(
            f"{var:12s} mean={np.nanmean(vals):.3f} std={np.nanstd(vals):.3f} "
            f"min={np.nanmin(vals):.3f} max={np.nanmax(vals):.3f}"
            f"nan={100*n_nan/n_total:.1f}"
        )

def plot_correlation_heatmap(df, title="Variable correlations"):
    """
    Correlation matrix across flattened variables. Used to see which
    variables might be independent or which are redundant.
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(corr, vmin=1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.colmuns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, label="correlation")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return corr

# ====================================================================
# Feature Engineering
# ====================================================================

