# Cancer Detection ML Pipeline

A production-ready machine learning pipeline for cancer detection using PyTorch, MLflow, and KServe, with a Streamlit web interface.

## Features

- PyTorch neural network for binary classification
- MLflow integration for experiment tracking
- Multiple synthetic datasets for robust training
- Streamlit web interface for model inference
- KServe deployment configuration
- Comprehensive model evaluation

## Project Structure

```
.
├── config.py         # Configuration settings
├── model.py          # Model architecture
├── data.py           # Data preparation
├── train.py          # Training script
├── app.py            # Streamlit web interface
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

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Deploy to KServe:

```bash
kubectl apply -f kserve.yaml
```

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

## Streamlit Interface

- Interactive feature input
- Real-time predictions
- Model information display
- Feature importance visualization

## Production Considerations

- Model validation
- Data versioning
- Monitoring
- Scalability
- Error handling
