from torch import Tensor as TorchTensor
from torch.nn import Module as TorchModule


class Loss:
    def __init__(
            self,
            loss_name: str,
            loss_params: dict[str, float | str]
        ):
        super().__init__()

        self.loss_name = loss_name
        self.loss_params = loss_params

    def get_criterion_torch(self) -> TorchModule:
        losses_dict = {
            'dice': DiceTorch,
        }
        if self.loss_name in losses_dict:
            return losses_dict[self.loss_name](**self.loss_params)
        raise KeyError(f"Loss name '{self.loss_name}' in config does not match designed losses.")


class DiceTorch(TorchModule):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            prediction: TorchTensor,
            ground_truth: TorchTensor,
            smooth: float=1e-4
        ) -> TorchTensor:
        intersection = (prediction * ground_truth).sum(dim=(2,3))
        union = prediction.sum(dim=tuple(range(2, prediction.dim()))) + ground_truth.sum(dim=tuple(range(2, prediction.dim())))
        dice_coefficient = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice_coefficient.mean()
