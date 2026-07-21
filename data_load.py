import os
import re
import requests
import xarray as xr
from collections import defaultdict

def extract_download_block(sh_file):
    with open(sh_file, "r", encoding="utf-8") as f:
        text = f.read()

    start_marker = "download_files=\"$(cat <<EOF"
    end_marker = "EOF"

    if start_marker not in text:
        raise ValueError("Could not find download_files block start")

    block = text.split(start_marker)[1]

    if end_marker not in block:
        raise ValueError("Could not find EOF end marker")

    block = block.split(end_marker)[0]

    return block


#TARGET_VARS = {"grFrazil", "pr", "prsn", "sic", "sit", "snd", "snoToIce", "strairx", "strairy", "streng"}
TARGET_VARS = {"gridspec"}

def parse_block(block):
    data = []

    for line in block.splitlines():
        if "http" not in line:
            continue

        parts = [p.strip().strip("'") for p in line.split()]

        if len(parts) < 4:
            continue

        filename = parts[0]
        url = parts[1]
        checksum_type = parts[2]
        checksum = parts[3]

        # variable is prefix before first "_"
        var = filename.split("_")[0]

        if var in TARGET_VARS:
            data.append({
                "var": var,
                "filename": filename,
                "url": url,
                "checksum": checksum
            })

    return data

def group_by_var(data):
    grouped = defaultdict(list)

    for item in data:
        grouped[item["var"]].append(item)

    return grouped


# Modify Cache for different models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data", "GFDL_ESM2M", "historical")
DEFAULT_SH_FILE = os.path.join(BASE_DIR, "data_bashes", "historical", "GFDL-ESM2M-hist.sh")
os.makedirs(CACHE_DIR, exist_ok=True)

def download_file(url, filename):
    path = os.path.join(CACHE_DIR, filename)

    if os.path.exists(path):
        return path

    print(f"Downloading: {filename}")

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return path

def download_all(grouped):
    local = defaultdict(list)

    for var, items in grouped.items():
        for item in items:
            path = download_file(item["url"], item["filename"])
            local[var].append(path)

    return local

def load_variable(files):
    return xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks={"time": 12}
    )

def build_dataset(local_files):
    datasets = {}

    for var, files in local_files.items():
        print(f"Loading {var}: {len(files)} files")
        datasets[var] = load_variable(files)

    return datasets

def load_cmip_from_sh(sh_file):
    sh_file = os.path.abspath(sh_file)

    if not os.path.exists(sh_file):
        raise FileNotFoundError(f"Could not find download script: {sh_file}")

    # 1. extract heredoc block
    block = extract_download_block(sh_file)

    # 2. parse variables
    data = parse_block(block)

    # 3. group by variable
    grouped = group_by_var(data)

    # 4. download (cached)
    local_files = download_all(grouped)

    # 5. load into xarray
    datasets = build_dataset(local_files)

    return datasets

if __name__ == "__main__":
    datasets = load_cmip_from_sh(DEFAULT_SH_FILE)