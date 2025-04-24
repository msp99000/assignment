
# Project: PyTorch + MLflow + KServe Deployment Workflow
# Goal: Simulate end-to-end architecture of training, logging with MLflow, and deploying via KServe.
# Target: Data Analytics SaaS use-case with future scale and robustness in mind.

# ----------------------
# STEP 1: Train and Log a Simple Iris Flower Classifier using MLflow
# ----------------------
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow.pytorch
import numpy as np

# Load and preprocess Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define simple classification model
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = IrisNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Train and log with MLflow
with mlflow.start_run():
    for epoch in range(10):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        mlflow.log_metric("loss", total_loss / len(train_loader), step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, artifact_path="iris-model")
    print("Iris model logged to MLflow!")

# ----------------------
# STEP 2: Model Artifact Explanation
# ----------------------
# MLflow logs the model in the structure below (this is what KServe needs):
# mlruns/<experiment-id>/<run-id>/artifacts/iris-model
# ├── MLmodel             <- Defines the model format and loader
# ├── model.pth           <- Serialized PyTorch model
# └── conda.yaml          <- Environment for deployment

# ----------------------
# STEP 3: Create a KServe InferenceService YAML
# ----------------------
# Save the following YAML as `inferenceservice.yaml`

# --- YAML START ---
# apiVersion: serving.kserve.io/v1beta1
# kind: InferenceService
# metadata:
#   name: iris-model
# spec:
#   predictor:
#     model:
#       modelFormat:
#         name: mlflow
#       storageUri: "s3://your-bucket/mlflow-models/iris-model"
# --- YAML END ---

# ----------------------
# STEP 4: System Architecture Diagram (text-based)
# ----------------------
#                     ┌────────────────────┐
#                     │  SaaS Dashboard UI │
#                     └─────────┬──────────┘
#                               │ REST API call
#                               ▼
#                         ┌────────────┐
#                         │  KServe    │ ◄─────┐
#                         │ (MLflow)   │       │
#                         └────┬───────┘       │
#                              │               │
#            InferenceRequest │               │ Model YAML
#                              ▼               │
#                      ┌──────────────┐        │
#                      │ IrisNet Class│        │
#                      │ via MLflow   │        │
#                      └──────────────┘        │
#                              ▲               │
#           Model URI from S3 ─┘               │
#                               ◄──────────────┘
#                  Deployment managed by Kubernetes

# ----------------------
# STEP 5: Final Notes for Interviewer
# ----------------------
# - MLflow simplifies model logging, versioning, and rollback.
# - KServe provides scalable, cloud-native model serving.
# - Suggest adding drift monitoring, model A/B testing, and auto-retraining pipelines.
# - Add GitOps (e.g., ArgoCD) for deploying model version YAMLs in a production ML pipeline.

# End of solution.
