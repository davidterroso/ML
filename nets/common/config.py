import yaml


class Config:
    def __init__(self, model: str) -> None:
        with open(f"nets/{model}/config.yaml", encoding='utf-8') as file:
            self.config = yaml.safe_load(file)

        self.dataset_name: str = self.config['dataset_name']

        self.model_name: str = self.config['model_name']
        self.model_parameters: dict[str, int | float | str] = self.config['model_parameters']

        self.optimizer_name: str = self.config['optimizer_name']
        self.optimizer_parameters: dict[str, int | float | str] = self.config['optimizer_parameters']

        self.loss_name: str = self.config['loss_name']
        self.loss_parameters: dict[str, int | float | str] = self.config['loss_parameters']

        self.epochs: int = self.config['epochs'] 
        self.batch_size: int = self.config['batch_size'] 
