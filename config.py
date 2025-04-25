from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class ModelConfig:
    input_size: int = 4
    hidden_size: int = 8
    output_size: int = 1
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 50
    n_samples: int = 1000
    n_features: int = 4
    n_informative: int = 3
    n_redundant: int = 0
    n_clusters_per_class: int = 1
    class_sep: float = 1.5

@dataclass
class MLflowConfig:
    experiment_name: str = "Cancer Detection Experiments"
    model_name: str = "cancer-classifier"
    tracking_uri: str = "http://localhost:5001"
    artifact_root: str = "./mlruns"

@dataclass
class StreamlitConfig:
    title: str = "Cancer Risk Assessment"
    description: str = "Enter patient features to assess cancer risk"
    feature_names: List[str] = field(default_factory=lambda: [
        "tumor_size",
        "cell_count",
        "nuclei_density",
        "mitosis_rate"
    ])
    feature_ranges: Dict[str, tuple] = field(default_factory=lambda: {
        "tumor_size": (0.0, 1.0),
        "cell_count": (0.0, 1.0),
        "nuclei_density": (0.0, 1.0),
        "mitosis_rate": (0.0, 1.0)
    })

@dataclass
class KServeConfig:
    model_format: str = "mlflow"
    storage_uri: str = "pvc://cancer-model"
    service_name: str = "cancer-classifier"
    mlflow_server: str = "http://localhost:5001"

# Create configuration instances
model_config = ModelConfig()
mlflow_config = MLflowConfig()
streamlit_config = StreamlitConfig()
kserve_config = KServeConfig() 