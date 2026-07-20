from typing import Dict, Union

import torch
from torch.optim import Adam, SGD

from nets.unet.torch.unet import UNet

def get_optimizer(
        optimizer_name: str,
        optimizer_params: Dict[str, Union[float, str]],
        model_parameters: torch.nn.Parameter,
    ):
    optimizers_dict = {
        'adam': Adam,
        'sgd': SGD,
    }
    if optimizer_name in optimizers_dict.keys():
        return optimizers_dict[optimizer_name](model_parameters, **optimizer_params)
    raise KeyError(f'No optimizer named {optimizer_name}.')


def get_model(
        model_name: str,
        model_params: Dict[str, Union[float, str]],
    ):
    models_dict = {
        'unet': UNet(**model_params),
    }
    if model_name in models_dict.keys():
        return models_dict[model_name]
    raise KeyError(f"Model name '{model_name}' in config does not match designed models.")


def run_train(params: Dict[str, Union[float, str, Dict[str, Union[float, str]]]]):

    model = get_model(params['model_name'], params['model_parameters'])

    optimizer = get_optimizer(params['optimizer'], params['optimizer_parameters'], model.parameters())
    print(optimizer)

    for epoch in range(1, params['epochs'] + 1):
        print(f"Epoch {epoch}/{params['epochs']} completed.")
    return