from torch.utils.data import DataLoader, Dataset

from nets.datasets.kvar_seg import KvarSeg


def get_dataset_torch(dataset_name: str) -> Dataset:
    datasets_dict = {
        "kvar_seg": KvarSeg,
    }
    if dataset_name in datasets_dict:
        return datasets_dict[dataset_name]()
    raise KeyError(f'No optimizer named {dataset_name}.')

def get_dataloader_torch(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, persistent_workers=True)
