# Sea Ice Concentration/Thickness ML Diagnostics

Investigating why coupled climate models (CMIP5) underestimate the observed
decline in Arctic sea ice, using ML on model output to identify where and
when model-observation errors are largest.

## Background

CMIP models have historically underestimated the rate of observed Arctic
sea ice loss. Proposed explanations include missing ice-sensitivity
feedbacks and unrepresented natural variability. This project compares
multiple CMIP5 models' sea ice output against each other (and eventually
against satellite observations) to explore patterns in where models agree,
disagree, and diverge from reality.

## Models

| Model | Experiments | Notes |
|---|---|---|
| GFDL-CM3 | piControl, historical* | NOAA-GFDL |
| GFDL-ESM2M | piControl, historical* | NOAA-GFDL |
| CMCC-CESM | piControl, historical* | CMCC |

\* historical runs in progress

## Variables

- `sic` — sea ice concentration
- `sit` — sea ice thickness
- `grFrazil`, `pr`, `prsn`, `snoToIce`, `strairx`, `strairy`, `streng` — process/thermodynamic variables (GFDL models)
- `ialb`, `sim`, `tsice`, `transix`, `transiy` — process variables (CMCC-CESM)
- `areacello`, `deptho`, `sftof` — fixed grid metadata (needed for area-weighted extent)

## Project structure

```
data_bashes/       ESGF wget download scripts (piControl / historical)
data/               Downloaded .nc files (gitignored -- see Reproducing below)
observations/       NSIDC or other obs data for comparison (gitignored)
images/             Figures / plots
data_load.py        Functions to parse wget scripts and load local .nc files into xarray
model.py            Full pipeline: load -> explore -> preprocess -> feature engineer -> train -> evaluate
```

## Reproducing the data

Data files are not committed to this repo (too large for git). To regenerate:

In data_load.py, change TARGET_VARS to select which variables you would like to download (to split the download into smaller chunks), then change 
CACHE_DIR and DEFAULT_SH_FILE to the folder you want to store the .nc variable files.

Downloaded files should be organized as described in `model.py`'s config
(model / experiment / variable folder structure).

## Status

- [x] Data loader for local .nc files (`data_load.py`)
- [x] Pipeline scaffold with exploratory visualization, preprocessing,
      feature engineering, modeling, and evaluation stages (`model.py`)
- [ ] Historical run data for all three models
- [ ] Observational (NSIDC) comparison data
- [ ] Area-weighted extent calculation using `areacello`
- [ ] Model-vs-observation error/residual as ML target

- [X] GDFL_CM3 piControl
- [X] GDFL_CM3 areacello
- [X] GDFL_CM3 historical
- [X] GDFL_ESM2M piControl 
- [X] GDFL_ESM2M areacello
- [] GDFL_ESM2M historical
- [X] CMCC_CESM piControl
- [] CMCC_CESM historical