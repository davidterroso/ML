import os

from torch.utils.data import Dataset
from torchvision.io import decode_image

from nets.datasets.common import create_folder, download_zip, unzip_dataset


def get_data(download_link: str, dest_folder: str="./tmp_data/"):
    create_folder(dest_folder)
    file_path = download_zip(download_link, dest_folder)
    unzip_dataset(file_path, dest_folder, rm=True)

class KvarSegTorch(Dataset):
    def __init__(self, data_path: str="./tmp_data/"):
        super().__init__()
        self.download_link = "https://datasets.simula.no/downloads/kvasir-seg.zip"
        get_data(self.download_link) 

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.ground_truth)
        image = decode_image(img_path)
        return image, index
