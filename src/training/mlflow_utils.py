import mlflow
import mlflow.pyfunc
from typing import Any
import logging
import yaml 
logger = logging.getLogger(__name__)

def load_mlflow_config(config_path: str = "configs/mlflow_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_tracking(config_path: str = "configs/mlflow_config.yaml"):
    config = load_mlflow_config(config_path)
    mlflow.set_tracking_uri(config["tracking_uri"])
    logger.info(f"mlflow tracking uri set to: {config['tracking_uri']}")

def setup_experiments(config_path: str = "configs/mlflow_config.yaml"):
    config = load_mlflow_config(config_path)
    mlflow.create_experiment(config['experiment']) if not mlflow.get_experiment_by_name(config['experiment']) else None
    logger.info(f"experiment ready: {config['experiment']}")


def start_run(run_name: str, config_path: str = "configs/mlflow_config.yaml"):
    config = load_mlflow_config(config_path)
    experiment = mlflow.get_experiment_by_name(config['experiment'])
    if experiment is None:
        raise ValueError(f"experiment '{config['experiment']}' not found. Run setup_experiments() first.")
    
    return mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=run_name
    )

def end_run():
    mlflow.end_run()


def log_config(config: dict[str, Any]):
    flat = _flatten_dict(config)
    mlflow.log_params(flat)
    logger.info(f"logged {len(flat)} config params to MLflow")

def log_metrics(metrics: dict[str, float], step: int):
    mlflow.log_metrics(metrics, step=step)

def log_artifact(local_path: str):
    mlflow.log_artifact(local_path)
    logger.info(f"artifact logged: {local_path}")


def register_model(run_id: str, artifact_path: str, model_name: str, config_path: str = "configs/mlflow_config.yaml"):
    config = load_mlflow_config(config_path)
    model_name = config["model_registry"]
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info(f"model registered: {model_name} v{result.version}")
    return result


def _flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items