import os
import shutil

from zipfile import ZipFile
from torch.utils.data import DataLoader, Dataset

from nets.datasets.kvar_seg import KvarSegTorch


def create_folder(folder: str="./tmp_data/") -> None:
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

def donwload_zip(url, dest_folder) -> str:
    print("Downloading data ...")
    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()

            with open(dest_folder, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
        return dest_folder + url.split(os.sep)[-1]
        print("Download finished successfully!")
    except Exception as e:
        raise f"Download failed: {e}"

def unzip_dataset(file_path: str, dest_folder: str)
    print("Extracting data ...")
    with ZipFile(file_path, 'r') as zObject:
        zObject.extractall(dest_folder)
    print("Extraction finished successfully!")
    if rm:
        os.remove(file_path)
        print(f"{file_path} deleted successfully!")

def get_dataset_torch(dataset_name: str) -> Dataset:
    datasets_dict = {
        "kvar_seg": KvarSegTorch,
    }
    if dataset_name in datasets_dict:
        return datasets_dict[dataset_name]()
    raise KeyError(f'No optimizer named {dataset_name}.')

def get_dataloader_torch(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, persistent_workers=True)
