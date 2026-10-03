from collections.abc import Iterator

from torch.nn import Parameter
from torch.optim import SGD, Adam
from torch.optim import Optimizer as TorchOptimizer


class Optimizer:
    def __init__(
            self,
            optimizer_name: str,
            optimizer_params: dict[str, int | float | str],
            model_params: Iterator[Parameter]
        ) -> None:

        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.model_params = model_params

    def get_optimizer_torch(self) -> TorchOptimizer:

        optimizers_dict = {
            'adam': Adam,
            'sgd': SGD,
        }

        if self.optimizer_name in optimizers_dict:
            return optimizers_dict[self.optimizer_name](self.model_params, **self.optimizer_params)

        raise ValueError(f'No optimizer named {self.optimizer_name}.')
