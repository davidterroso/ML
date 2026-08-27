import torch

from nets.unet.torch.unet import UNet


def get_model_torch(
        model_name: str,
        model_params: dict[str, float | str],
    ) -> torch.nn.Module:
    models_dict = {
        'unet': UNet,
    }
    if model_name in models_dict:
        return models_dict[model_name](**model_params)
    raise KeyError(f"Model name '{model_name}' in config does not match designed models.")