import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import glob

DATA_DIR = "GDFL_CM3"

VAR_PATTERNS = {
    "sic": "sic_OImon_GFDL-CM3_piControl_r1i1p1_*.nc",
    "sit": "sit_OImon_GFDL-CM3_piControl_r1i1p1_*.nc"
}

