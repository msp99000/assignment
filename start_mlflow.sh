#!/bin/bash

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Start MLflow server with SQLite backend
mlflow server \
    --backend-store-uri sqlite:///mlruns/mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5001 