import mlflow
from src.training.mlflow_utils import setup_experiments

mlflow.set_tracking_uri("http://localhost:5000")
setup_experiments()
print("Done. Check localhost:5000")