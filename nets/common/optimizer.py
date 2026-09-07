from collections.abc import Iterator

import torch
from torch.optim import SGD, Adam


def get_optimizer_torch(
        optimizer_name: str,
        optimizer_params: dict[str, float | str],
        model_parameters: Iterator[torch.nn.Parameter],
    ) -> torch.optim.Optimizer:
    optimizers_dict = {
        'adam': Adam,
        'sgd': SGD,
    }
    if optimizer_name in optimizers_dict:
        return optimizers_dict[optimizer_name](model_parameters, **optimizer_params)
    raise KeyError(f'No optimizer named {optimizer_name}.')
