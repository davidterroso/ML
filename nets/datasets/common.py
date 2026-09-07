import os
import shutil
from zipfile import ZipFile

import requests

from nets.datasets.cert import build_ca_bundle


def create_folder(folder: str="./tmp_data/") -> None:
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

def download_zip(url, dest_folder) -> str:
    print("Downloading data ...")
    CA_BUNDLE = build_ca_bundle()
    try:
        with requests.get(url, stream=True, verify=CA_BUNDLE, timeout=30) as response:
            response.raise_for_status()
            file_name = dest_folder + url.split(os.sep)[-1]
            with open(file_name, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
        print("Download finished successfully!")
        return file_name
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}") from e

def unzip_dataset(file_path: str, dest_folder: str, rm: bool=True) -> str:
    print("Extracting data ...")
    with ZipFile(file_path, 'r') as zObject:
        zObject.extractall(dest_folder)
    print("Extraction finished successfully!")
    if rm:
        os.remove(file_path)
        print(f"{file_path} deleted successfully!")

    unzipped_folder = next(entry.name for entry in os.scandir(dest_folder) if entry.is_dir())

    return str(dest_folder + unzipped_folder + os.sep)
