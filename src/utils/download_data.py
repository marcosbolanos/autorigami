from __future__ import annotations

import os
import requests  # type: ignore[reportMissingModuleSource]
from typing import Mapping

from tqdm import tqdm  # type: ignore[reportMissingModuleSource]

from .. import PROJECT_ROOT

# Use absolute path based on script location to ensure consistent data directory
# This will always point to the project's data folder regardless of where the script is called from
DATA_DIR = os.path.join(str(PROJECT_ROOT), "data", "external")
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Data directory: {DATA_DIR}")


def download_datasets(datasets: Mapping[str, str]) -> None:
    for name, url in datasets.items():
        filepath = os.path.join(DATA_DIR, name)
        if not os.path.exists(filepath):
            print(f"Donwloading {name}...")

            # Stream download with progress bar
            response = requests.get(url, stream=True, verify=False)
            total_size = int(response.headers.get("content-length") or 0)

            with (
                open(filepath, "wb") as f,
                tqdm(
                    desc=name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f"{name} downloaded to {filepath}")
        else:
            print(f"{name} already present.")


def extract_files(datasets: Mapping[str, str]) -> None:
    if __name__ == "__main__":
        _models = {
            "teapot": "https://upload.wikimedia.org/wikipedia/commons/9/93/Utah_teapot_%28solid%29.stl",
        }
        download_datasets(_models)
