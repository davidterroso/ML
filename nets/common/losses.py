import torch
from torch import nn


class TorchLoss(nn.Module):
    def __init__(self):
        super().__init__()


class DiceTorch(TorchLoss):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            prediction: torch.Tensor,
            ground_truth: torch.Tensor,
            smooth: 1e-4
        ) -> torch.Tensor:
        intersection = (prediction * ground_truth).sum(dim=(2,3))
        union = prediction.sum(dim=tuple(range(2, prediction.dim()))) + ground_truth.sum(dim=tuple(range(2, prediction.dim())))
        dice_coefficient = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice_coefficient.mean()

def get_criterion_torch(
        loss_name: str,
        loss_params: dict[str, float | str]
    ) -> TorchLoss:
    losses_dict = {
        'dice': DiceTorch,
    }
    if loss_name in losses_dict:
        return losses_dict[loss_name](**loss_params)
    raise KeyError(f"Loss name '{loss_name}' in config does not match designed losses.")
