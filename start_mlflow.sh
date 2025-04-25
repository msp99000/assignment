#!/bin/bash

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Start MLflow server with local backend store and artifact root
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlruns/mlflow.db \
    --default-artifact-root ./mlruns 