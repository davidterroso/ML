import os

from torch.utils.data import Dataset
from torchvision.io import decode_image


class KvarSeg(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.ground_truth)
        image = decode_image(img_path)
        return image, index