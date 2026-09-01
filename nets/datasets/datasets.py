from torch.utils.data import DataLoader, Dataset

from nets.datasets.kvar_seg import KvarSegTorch

DATASETS_DICT = {
    "kvar_seg": KvarSegTorch,
}

def get_dataset_torch(dataset_name: str) -> Dataset:
    if dataset_name in DATASETS_DICT:
        return DATASETS_DICT[dataset_name]()
    raise KeyError(f'No optimizer named {dataset_name}.')


def get_dataloader_torch(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, persistent_workers=True)
