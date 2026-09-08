from datasets.datasets import Dataset

from nets.common.config import Config
from nets.common.losses import Loss
from nets.common.models import Model
from nets.common.optimizer import Optimizer


def run_train(config: Config) -> None:

    model = Model(
        model_name=config.model_name,
        model_parameters=config.model_parameters
    ).get_model_torch()

    optimizer = Optimizer(
        optimizer_name=config.optimizer_name,
        optimizer_params=config.optimizer_parameters
    ).get_optimizer_torch()

    criterion = Loss(
        loss_name=config.loss_name,
        loss_params=config.loss_parameters
    ).get_criterion_torch()

    dataloader = Dataset(
        dataset_name=config.dataset_name,
        batch_size=config.batch_size
    ).get_dataloader_torch()


    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0
        for batch in dataloader:
            inputs, ground_truth = batch
            optimizer.zero_grad()
            prediction = model(inputs)
            batch_loss = criterion(prediction, ground_truth)
            epoch_loss += batch_loss
        print(f"Epoch {epoch}/{config.epochs} completed.")
