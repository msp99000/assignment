# Cancer Detection ML Pipeline

A production-ready machine learning pipeline for cancer detection using PyTorch, MLflow, and KServe, with a Streamlit web interface (incomplete).

## Features

- PyTorch neural network for binary classification
- MLflow integration for experiment tracking
- Multiple synthetic datasets for robust training
- KServe deployment for model serving
- Local development and testing setup
- Comprehensive model evaluation

## Prerequisites

- Python 3.8+
- Minikube
- kubectl
- Docker
- MLflow
- PyTorch
- scikit-learn

## Project Structure

```
.
├── config.py         # Configuration settings
├── model.py          # Model architecture
├── data.py           # Data preparation
├── train.py          # Training script
├── kserve.yaml       # KServe deployment config
├── start_mlflow.sh   # MLflow server startup script
├── setup_local.sh    # Local environment setup script
├── test_predictions.py # Prediction testing script
└── requirements.txt  # Dependencies
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MLflow Server

In one terminal:

```bash
chmod +x start_mlflow.sh
./start_mlflow.sh
```

### 3. Train the Model

In another terminal:

```bash
python train.py
```

This will:

- Generate synthetic cancer dataset
- Train the model
- Save the model to MLflow
- Log metrics and parameters

### 4. Set Up Local Environment with KServe

In a new terminal:

```bash
chmod +x setup_local.sh
./setup_local.sh
```

This script will:

- Start Minikube
- Install KServe
- Create necessary Kubernetes resources
- Deploy the model
- Set up port forwarding

### 5. Test Predictions

Once the setup is complete, you can test predictions:

```bash
python test_predictions.py
```

This will make predictions on example data and display the results.

## Model Architecture

- Input layer: 4 features (tumor size, cell count, nuclei density, mitosis rate)
- Hidden layer: 8 units with batch normalization and ReLU activation
- Dropout: 0.2
- Output layer: 1 unit with sigmoid activation

## MLflow Integration

- Tracks training metrics (loss, accuracy, precision, recall, F1)
- Logs model parameters
- Saves best model based on test accuracy
- Model registry for versioning

## KServe Deployment

- Model served through KServe
- Local development setup with Minikube
- ConfigMap-based model storage
- Port forwarding for local access

## Troubleshooting

1. If Minikube fails to start:

   ```bash
   minikube delete
   minikube start --memory=4096 --cpus=2
   ```

2. If KServe deployment fails:

   ```bash
   kubectl delete -f kserve.yaml
   kubectl apply -f kserve.yaml
   ```

3. If predictions fail:
   - Check if MLflow server is running
   - Verify KServe deployment status
   - Check port forwarding

## Future Enhancements

- Streamlit web interface (incomplete)
- Automated retraining pipeline
- Model monitoring and drift detection
- Production deployment guide
