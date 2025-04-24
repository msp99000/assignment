from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    input_size: int = 4
    hidden_size: int = 16
    output_size: int = 3
    learning_rate: float = 0.01
    batch_size: int = 16
    epochs: int = 10

@dataclass
class MLflowConfig:
    experiment_name: str = "iris-classification"
    model_name: str = "iris-model"
    tracking_uri: str = "http://localhost:5000"  # Default MLflow tracking URI

@dataclass
class KServeConfig:
    model_format: str = "mlflow"
    storage_uri: str = "s3://your-bucket/mlflow-models/iris-model"  # Update this with your S3 bucket
    service_name: str = "iris-model"

# Create configuration instances
model_config = ModelConfig()
mlflow_config = MLflowConfig()
kserve_config = KServeConfig() 