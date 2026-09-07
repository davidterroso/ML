import argparse

import yaml

from nets.common.train_jax import run_train as run_train_jax
from nets.common.train_torch import run_train as run_train_torch


def load_params(model: str) -> dict[str, int | float | str | dict[str, int | float | str]]:
    with open(f"nets/{model}/parameters.yaml", encoding='utf-8') as file:
        params = yaml.safe_load(file)
    return params

def main():
    parser = argparse.ArgumentParser("Entry point to model training.")
    parser.add_argument('model_name',
                        type=str,
                        help="Name of the network to be trained (e.g. 'unet').")
    parser.add_argument('framework',
                        type=str,
                        help="Name of the DL framework to be used. Can be 'torch' or 'jax'")
    args = parser.parse_args()
    model_name = args.model_name
    params = load_params(model_name)
    if args.framework == 'jax':
        run_train_jax(params)
    elif args.framework == 'torch':
        run_train_torch(params)
    else:
        raise ValueError('Unknown DL framework selected.')

    print(f"Model '{model_name}' finished training.")

if __name__ == "__main__":
    main()
