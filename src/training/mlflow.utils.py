import mlflow
import mlflow.pyfunc
from typing import Any
import logging

logger = logging.getLogger(__name__)


def setup_experiments(experiment_name:str, tracking_uri: str = "http://localhost:5000"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(experiment_name) if not mlflow.get_experiment_by_name(experiment_name) else None
    logger.info(f"experiment ready: {experiment_name}")


def start_run(experiment_name: str, run_name: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"experiment '{experiment_name}' not found. Run setup_experiments() first.")
    
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


def register_model(run_id: str, artifact_path: str, model_name: str):
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