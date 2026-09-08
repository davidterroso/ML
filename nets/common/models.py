from typing import cast

from torch.nn import Module as TorchModule


class Model:
    def __init__(
            self,
            model_name: str,
            model_parameters: dict[str, int | float | str]
        ) -> None:

        self.model_name = model_name
        self.model_parameters = model_parameters


    def get_model_torch(self) -> TorchModule:

        from nets.unet.torch.unet import UNet, UNetParams

        models_dict = {
            'unet': UNet,
        }

        self.model_parameters = cast(UNetParams, self.model_parameters)

        if self.model_name in models_dict:
            return models_dict[self.model_name](**self.model_parameters)

        raise KeyError(f"Model name '{self.model_name}' in config does not match designed models.")
