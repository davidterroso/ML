from tqdm import tqdm

from nets.common.config import Config
from nets.common.losses import Loss
from nets.common.models import Model
from nets.common.optimizer import Optimizer
from nets.datasets.datasets import Dataset


def run_train(config: Config) -> None:

    model = Model(
        model_name=config.model_name,
        model_parameters=config.model_parameters
    ).get_model_torch()

    optimizer = Optimizer(
        optimizer_name=config.optimizer_name,
        optimizer_params=config.optimizer_parameters,
        model_params=model.parameters()
    ).get_optimizer_torch()

    criterion = Loss(
        loss_name=config.loss_name,
        loss_params=config.loss_parameters
    ).get_criterion_torch()

    train_dataset = Dataset(
        train=True,
        dataset_name=config.dataset_name,
        batch_size=config.batch_size
    )
    train_dataset.build_dataset_torch()
    train_dataloader = train_dataset.get_dataloader_torch()

    val_dataset = Dataset(
        train=False,
        dataset_name=config.dataset_name,
        batch_size=config.batch_size
    )
    val_dataset.build_dataset_torch()
    val_dataloader = val_dataset.get_dataloader_torch()


    for epoch in range(1, config.epochs + 1):

        epoch_loss = 0

        with tqdm(total=len(train_dataloader),
                  desc=f"Training (Epoch {epoch})",
                  unit="images",
                  ascii="░▒█",
                  colour="green"
                ) as pbar:

            for batch in train_dataloader:
                print(batch)
                break
                inputs, ground_truth = batch

                optimizer.zero_grad()

                prediction = model(inputs)

                loss = criterion(prediction, ground_truth)
                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(len(batch))

            
        with tqdm(total=len(val_dataloader),
                  desc=f"Validation (Epoch {epoch})",
                  unit="images",
                  ascii="░▒█",
                  colour="red"
                ) as pbar:
            for batch in val_dataloader:
                print(batch)
                break
                pbar.update(1)

    print(f"Model '{config.model_name}' has completed training.")
