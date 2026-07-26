import torch
from torch.optim import SGD, Adam

from nets.unet.torch.unet import UNet
from nets.common.losses import DiceTorch, TorchLoss


def get_optimizer(
        optimizer_name: str,
        optimizer_params: dict[str, float | str],
        model_parameters: torch.nn.Parameter,
    ) -> torch.optim.Optimizer:
    optimizers_dict = {
        'adam': Adam,
        'sgd': SGD,
    }
    if optimizer_name in optimizers_dict:
        return optimizers_dict[optimizer_name](model_parameters, **optimizer_params)
    raise KeyError(f'No optimizer named {optimizer_name}.')


def get_model(
        model_name: str,
        model_params: dict[str, float | str],
    ) -> torch.nn.Module:
    models_dict = {
        'unet': UNet,
    }
    if model_name in models_dict:
        return models_dict[model_name](**model_params)
    raise KeyError(f"Model name '{model_name}' in config does not match designed models.")


def get_criterion(
        loss_name: str,
        loss_params: dict[str, float | str]
    ) -> TorchLoss:
    losses_dict = {
        'dice': DiceTorch,
    }
    if loss_name in losses_dict:
        return losses_dict[loss_name](**loss_params)
    raise KeyError(f"Loss name '{loss_name}' in config does not match designed losses.")


def run_train(params: dict[str, float | str | dict[str, float | str]]):

    model = get_model(params['model_name'], params['model_parameters'])
    optimizer = get_optimizer(params['optimizer'], params['optimizer_parameters'], model.parameters())
    criterion = get_criterion(params['loss_function'])

    for epoch in range(1, params['epochs'] + 1):
        epoch_loss = 0
        for batch in batches:
            optimizer.zero_grad()
            prediction = model(inputs)
            batch_loss = criterion(prediction, ground_truth)
            epoch_loss += batch_loss
        print(f"Epoch {epoch}/{params['epochs']} completed.")
