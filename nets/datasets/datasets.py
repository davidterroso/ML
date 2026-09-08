from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from nets.datasets.kvasir_seg import KvasirSegTorch

DATASETS_DICT = {
    "kvar_seg": KvasirSegTorch,
}


class Dataset:
    def __init__(self,
                 dataset_name: str,
                 batch_size: int
        ) -> None:

        self.dataset_name = dataset_name
        self.batch_size = batch_size

    def get_dataset_torch(self) -> TorchDataset:
        if self.dataset_name in DATASETS_DICT:
            self.dataset = DATASETS_DICT[self.dataset_name]() 
            return self.dataset
        raise KeyError(f'No optimizer named {self.dataset_name}.')


    def get_dataloader_torch(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=-1
        )
