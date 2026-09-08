from nets.common.config import Config


def run_train(config: Config):
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs} completed.")
