import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from nets.datasets.kvasir_seg import KvasirSegTorch

TORCH_DATASETS_DICT = {
    "kvasir_seg": KvasirSegTorch,
}


class Dataset:
    def __init__(self,
                 dataset_name: str,
                 batch_size: int,
                 train: bool
        ) -> None:

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.train = train

    def build_dataset_torch(self) -> None:
        if self.dataset_name in TORCH_DATASETS_DICT:
            self.dataset: TorchDataset = TORCH_DATASETS_DICT[self.dataset_name](self.train) 
        else:
            raise ValueError(f'No dataset named {self.dataset_name}.')


    def get_dataset_torch(self) -> TorchDataset:
        if self.dataset is None:
            self.build_dataset_torch()
        return self.dataset
        

    def get_dataloader_torch(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=os.cpu_count() or 0
        )
