# MLflow + KServe Iris Classification

A production-ready machine learning pipeline for Iris flower classification using PyTorch, MLflow, and KServe.

## Features

- PyTorch neural network with batch normalization and dropout
- MLflow integration for experiment tracking and model registry
- KServe deployment configuration
- Comprehensive model evaluation
- Example inference script

## Project Structure

```
.
├── config.py         # Configuration settings
├── model.py          # Model architecture
├── data.py           # Data preparation
├── train.py          # Training script
├── inference.py      # Inference example
├── kserve.yaml       # KServe deployment config
└── requirements.txt  # Dependencies
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start MLflow server:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

## Usage

1. Train the model:

```bash
python train.py
```

2. Run inference:

```bash
python inference.py
```

3. Deploy to KServe:

```bash
kubectl apply -f kserve.yaml
```

## Model Architecture

- Input layer: 4 features (sepal length, sepal width, petal length, petal width)
- Hidden layer: 16 units with batch normalization and ReLU activation
- Dropout: 0.2
- Output layer: 3 classes (Setosa, Versicolor, Virginica)

## MLflow Integration

- Tracks training metrics (loss, accuracy)
- Logs model parameters
- Saves best model based on test accuracy
- Model registry for versioning

## KServe Deployment

- MLflow model format
- Resource limits configured
- Scalable deployment

## Production Considerations

- Model validation
- Data versioning
- Monitoring
- Scalability
- Error handling
