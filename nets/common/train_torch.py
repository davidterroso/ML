from datasets.datasets import get_dataloader_torch as get_dataloader
from datasets.datasets import get_dataset_torch as get_dataset

from nets.common.losses import get_criterion_torch as get_criterion
from nets.common.models import get_model_torch as get_model
from nets.common.optimizer import get_optimizer_torch as get_optimizer


def run_train(params: dict[str, int | float | str | dict[str, int | float | str]]):

    model = get_model(str(params['model_name']), params['model_parameters'])
    optimizer = get_optimizer(str(params['optimizer']), params['optimizer_parameters'], model.parameters())
    criterion = get_criterion(str(params['loss_function']), loss_params=params['loss_params'])
    dataset = get_dataset(params['dataset_name'])
    dataloader = get_dataloader(dataset, params['batch_size'])

    for epoch in range(1, int(params['epochs']) + 1):
        epoch_loss = 0
        for batch in dataloader:
            inputs, ground_truth = batch
            optimizer.zero_grad()
            prediction = model(inputs)
            batch_loss = criterion(prediction, ground_truth)
            epoch_loss += batch_loss
        print(f"Epoch {epoch}/{params['epochs']} completed.")
