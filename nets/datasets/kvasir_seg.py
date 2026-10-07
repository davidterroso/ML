import os

from pandas import Series
from torch.utils.data import Dataset
from torchcodec.decoders._image_decoders import decode_jpeg
from torchvision.transforms import v2

from nets.datasets.common import create_folder, download_zip, unzip_dataset


def _process_data(
        file_path: str,
    ) -> tuple[Series, Series]:

    images_df = Series([f for f in os.listdir(str(file_path + os.sep + "images" + os.sep))])
    masks_df = Series([f for f in os.listdir(str(file_path + os.sep + "masks" + os.sep))])

    return images_df, masks_df

def _init_data(
        data_folder: str,
        download_link: str,
    ) -> tuple[str, Series, Series]:
    create_folder(data_folder)
    zip_path = download_zip(download_link, data_folder)
    unzipped_folder = unzip_dataset(zip_path, data_folder, rm=True)

    images_df, masks_df = _process_data(unzipped_folder)

    return unzipped_folder, images_df, masks_df

class KvasirSegTorch(Dataset):
    def __init__(self, train: bool, device: str="cpu"):
        super().__init__()
        self.train = train

        self.device = device

        self.data_folder = f".{os.sep}tmp_data{os.sep}"
        self.download_link = "https://datasets.simula.no/downloads/kvasir-seg.zip"

        self.folder_path, self.images_df, self.masks_df = _init_data(self.data_folder, self.download_link)
        self.transform = v2.Compose([
            v2.Resize(size=[3, 256, 256])
        ])

    def __len__(self):
        return len(self.masks_df)

    def __getitem__(self, index): 
        assert self.images_df[index] == self.masks_df[index]

        img_path = str(self.folder_path + "images" + os.sep + self.images_df[index])
        mask_path = str(self.folder_path + "masks" + os.sep + self.masks_df[index])

        image_jpg = decode_jpeg(img_path, device=self.device)
        image = self.transform(image_jpg)

        mask_jpg = decode_jpeg(mask_path, device=self.device)
        mask = self.transform(mask_jpg)

        return image, mask, index
