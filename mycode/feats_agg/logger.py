from abc import ABC

from pytorch_lightning.loggers import Logger
from clearml import Task
import wandb
import mlflow


class CustomLogger(Logger, ABC):
    def __init__(self, loggers_params):
        super().__init__()
        self.mlflow_run = None
        self.clearml_task = None
        self.loggers = []
        self.loggers_params = loggers_params
        self.log_dir = None
        self.mlflow_experiment_name = None
        self.wandb_project = None

        for logger_params in self.loggers_params:
            logger_name = logger_params.pop("name", None)
            if logger_name:
                self.loggers.append(logger_name)
                if logger_name == "mlflow":
                    self.mlflow_experiment_name = logger_params.get("mlflow_experiment_name")
                elif logger_name == "wandb":
                    self.wandb_project = logger_params.get("wandb_project")
                elif logger_name == "file":
                    self.log_dir = logger_params.get("log_dir")
                elif logger_name == "clearml":
                    task_name = logger_params.pop("task_name", "My Task")
                    self.clearml_task = Task.init(task_name=task_name, **logger_params)

        if not self.log_dir:
            self.log_dir = "logs/"

    def log_metrics(self, metrics, step=None):
        for logger_name in self.loggers:
            if logger_name == "mlflow":
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            elif logger_name == "wandb":
                for key, value in metrics.items():
                    wandb.log({key: value}, step=step)
            elif logger_name == "clearml":
                for key, value in metrics.items():
                    self.clearml_task.logger.report_scalar(title=key, series="metrics", value=value, iteration=step)

    def log_config(self, config):
        for logger_name in self.loggers:
            if logger_name == "mlflow":
                mlflow.log_params(config)
            elif logger_name == "wandb":
                wandb.config.update(config)
            elif logger_name == "clearml":
                self.clearml_task.set_model_config(config_dict=config)

    def setup(self):
        for logger_name, logger_params in zip(self.loggers, self.loggers_params):
            if logger_name == "mlflow":
                mlflow.set_experiment(logger_params.get("mlflow_experiment_name"))
                mlflow.start_run(**logger_params)
                self.mlflow_run = mlflow.active_run()
            elif logger_name == "wandb":
                wandb.init(**logger_params)
            elif logger_name == "clearml":
                pass  # ClearML is initialized during __init__

    def teardown(self, stage):
        for logger_name in self.loggers:
            if logger_name == "mlflow":
                mlflow.end_run()
            elif logger_name == "wandb":
                wandb.join()
            elif logger_name == "clearml":
                self.clearml_task.close()

    @property
    def experiment(self):
        return None

    @property
    def name(self):
        return "CustomLogger"


# Example usage
# loggers_params = [
#     {"name": "mlflow", "mlflow_experiment_name": "My Experiment"},
#     {"name": "wandb", "wandb_project": "My Project"},
#     {"name": "file", "log_dir": "logs/"},
#     {"name": "clearml", "task_name": "My Task", "project_name": "My Project", "auto_connect": True}
# ]
# logger = CustomLogger(loggers_params=loggers_params)
# trainer = pl.Trainer(logger=logger)

# Your PyTorch Lightning training code
