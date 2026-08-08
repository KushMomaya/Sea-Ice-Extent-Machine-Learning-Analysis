import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr # Specifally good for working with NetCDF files
import os
import glob # Pathname pattern matching
from sklearn.decomposition import PCA

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

TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)
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
    return xr.open_mfdataset(files, combine="by_coords", chunks={"time": 12}, decode_times=TIME_CODER, data_vars="minimal", coords="minimal", compat="override")
    
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
            fx_datasets.append(xr.open_dataset(files[0], decode_times=TIME_CODER))
    
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
    return xr.open_mfdataset(files, combine="by_coords", decode_times=TIME_CODER, data_vars="minimal", coords="minimal", compat="override")

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
        if not np.issubdtype(vals.dtype, np.number):
            print(f"{var:12s} dtype={vals.dtype} (non-numeric, skipping statistics)")
            continue
        n_nan = np.isnan(vals).sum()
        n_total = vals.size
        print(
            f"{var:12s} mean={np.nanmean(vals):.3f} std={np.nanstd(vals):.3f} "
            f"min={np.nanmin(vals):.3f} max={np.nanmax(vals):.3f}"
            f" nan={100*n_nan/n_total:.1f}"
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
# Feature Engineering + Dimensionality Reduction
# ====================================================================


def subset_hemisphere(ds):
    """
    return only the data corresponding to the southern hemisphere (latitude below 0/equator)
    """
    return ds.where(ds["lat"] < 0, drop=True)

def compute_area_weighted_extent(sic_da, area_da, threshold=0.15):
    """
    Standard Sea Ice Extent Calculation: sum grid-cell area where sic >= threshold per timestep.
    """
    ice_mask = (sic_da >= threshold).astype(float)
    extent = (ice_mask * area_da).sum(dim=[d for d in sic_da.dims if d != "time"])
    return extent

def flatten_to_dataframe(ds, variables=None, model_name=None, experiment=None):
    """
    Convert an xr.Dataset variables into a pandas dataframe:
    One row per (time, spatial cell), one column per variable.
    """
    vars = variables or list(ds.data_vars)
    df = df[variables].to_dataframe().reset_index()
    df = df.dropna(subset=variables, how="all")
    if model_name:
        df["model"] = model_name
    if experiment:
        df["experiment"] = experiment
    return df

def add_temporal_features(df, time_col="time"):
    """
    Add seasonal_cycle and related temporal features.
    Circular month encoding using sin/cos to prevent Jan and Dec
    from being 11 months apart. 
    """
    df = df.copy()
    dt = pd.to_datetime(df[time_col])
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    df["year"] = dt.dt.year
    return df

def add_lag_features(df, target_col, group_cols=("lat", "lon"), lag=1):
    """
    Add a lagged feature of the target var (sic) as a feature.
    Used as a persistence baseline for a predictor to beat in terms
    of accuracy.
    """
    df = df.sort_values("time")
    df[f"{target_col}_lag{lag}"] = df.groupby(list(group_cols))[target_col].shift(lag)
    return df

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def pca_on_field(da, n_components=10):
    """
    da: DataArray for one var with dims (time, y, x) for one model (e.g. GFDL_CM3 sic)
    Reshapes da into a 2D matrix (time, n_cells) for pca. PCA finds the recurring spatial
    patterns that reconstruct majority of the variability in the full field. Also called
    Empirical Orthogonal Function Analysis which refers to PCA applied to (time x space)
    fields. 
    """
    stacked = da.stack(cell=[d for d in da.dims if d != "time"])
    values = stacked.values
    
    valid_mask = ~np.isnan(values).any(axis=0)
    values_valid = values[:, valid_mask]
    
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pc_scores = pca.fit_transform(values_valid)
    
    print(f"Explained variance by component: {pca.explained_variance_ratio_[:5]}")
    print(f"Cumulative (first in {n_components}): {pca.explained_variance_ratio_.sum():.2%}")
    
    return pca, pc_scores, valid_mask, stacked.coords

def plot_explained_variance(pca):
    """
    Plot of how much variance each additional component adds.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("number of componenets")
    plt.ylabel("cumulative explained variance")
    plt.title("PCA explained variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_eof_pattern(pca, component_idx, valid_mask, coords, title = None):
    """
    Reshapes PCA component back onto spatial grid and plots it. 
    """
    if component_idx >= pca.n_components_:
        raise ValueError(
            f"component_idx={component_idx} out of range, pca was fit with only {pca.n_components_} components"
        )
    full_loadings = np.full(valid_mask.shape, np.nan)
    full_loadings[valid_mask] = pca.components_(component_idx)
    
    pattern = xr.DataArray(full_loadings, dims=["cell"], coords={"cell": coords["cell"]})
    pattern = pattern.unstack("cell")
 
    variance_pct = pca.explained_variance_ratio_[component_idx] * 100
 
    fig, ax = plt.subplots(figsize=(7, 5))
    pattern.plot(ax=ax, cmap="RdBu_r", center=0)
    ax.set_title(title or f"EOF {component_idx + 1} ({variance_pct:.1f}% variance explained)")
    plt.tight_layout()
    plt.show()
 
    return pattern

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

def train_test_split(df, val_frac=0.15, test_frac=0.15):
    """
    Splits data into a train test validation split. Is NOT random split
    because the data is structured chronologically. If it was random then
    the training process could use data from after the test period.
    """
    times = np.sort(df["time"].unique())
    n = len(times)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    
    train_times = times[:, n - n_val - n_test]
    val_times = times[n - n_val - n_test: n - n_test]
    test_times = times[n - n_test:]
    
    train_df = df[df["time"].isin(train_times)]
    val_df = df[df["time"].isin(val_times)]
    test_df = df[df["time"].isin(test_times)]
    
    return train_df, val_df, test_df

# ============================================================
# Model Definition + Evaluation
# ============================================================

def climatology_baseline(train_df, test_df, target_col):
    """
    Predict test values using the historical mean for that calendar month,
    computer from train data.
    """
    monthly_clim = train_df.groupby("month")[target_col].mean()
    return test_df["month"].map(monthly_clim)

def build_and_train_mlmodel(x_train, y_train):
    """
    
    """



def main():
    data, fx = load_all_data()
    
    pi_ds = data.get("GFDL_CM3", {}).get("piControl")
    hist_ds = data.get("GFDL_CM3", {}).get("historical")
    
    summarize_dataset(pi_ds, "GFDL_CM3/piControl")
    summarize_dataset(pi_ds, "GFDL_CM3/historical")
    
    print("\nRunning exploratory plots...")
    run_eda_plots(data, fx)
    
    # Bias Calculation: model vs observation comparison for Antarctic/southern hemisphere
    
    obs_extent = load_obs_extent()
    
    sic_south = subset_hemisphere(hist_ds[TARGET_VAR])
    area_south = subset_hemisphere(fx["GFDL_CM3"]["areacello"])
    
    model_extent = compute_area_weighted_extent(sic_south, area_south)
    aligned = align_model_and_obs_extent(model_extent, obs_extent)
    aligned = compute_extent_bias(aligned)
    plot_extents(aligned)
    
    hist_south = subset_hemisphere(hist_ds)
    process_vars = [v for v in MODEL_VARS["GFDL_CM3"] if v != TARGET_VAR]
    process_df = flatten_to_dataframe(hist_south, "GFDL_CM3", experiment="historical")
    process_df = process_df.groupby("time")[process_vars].mean().reset_index()
    