from torch.optim import SGD, Adam
from torch.optim import Optimizer as TorchOptimizer


class Optimizer:
    def __init__(
            self,
            optimizer_name: str,
            optimizer_params: dict[str, int | float | str]
        ) -> None:

        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params

    def get_optimizer_torch(self) -> TorchOptimizer:

        optimizers_dict = {
            'adam': Adam,
            'sgd': SGD,
        }

        if self.optimizer_name in optimizers_dict:
            return optimizers_dict[self.optimizer_name](self.optimizer_params, **self.optimizer_params)

        raise KeyError(f'No optimizer named {self.optimizer_name}.')
