import torch
import mlflow
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from config import mlflow_config, streamlit_config
from model import create_model

def load_local_model(model_path, device='cpu'):
    """Load model from local checkpoint"""
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_mlflow_model():
    """Load the latest version of the model from MLflow"""
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    model = mlflow.pytorch.load_model(f"models:/{mlflow_config.model_name}/latest")
    model.eval()
    return model

def predict(model, features):
    """Make predictions using the loaded model"""
    # Convert features to tensor
    features = torch.tensor(features, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        model.eval()
        output = model(features)
        probability = output.numpy()
    
    return probability

def main():
    # Example features (tumor_size, cell_count, nuclei_density, mitosis_rate)
    example_features = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Low risk
        [8.3, 7.9, 6.6, 5.8],  # High risk
        [6.9, 5.0, 4.2, 3.5]   # Medium risk
    ])
    
    # Try loading from MLflow first, then fall back to local model
    try:
        print("Attempting to load model from MLflow...")
        model = load_mlflow_model()
        print("Model loaded successfully from MLflow!")
    except Exception as e:
        print(f"Failed to load from MLflow: {e}")
        print("Attempting to load latest local checkpoint...")
        checkpoint_dir = 'checkpoints'
        if os.path.exists(checkpoint_dir):
            latest_checkpoint = max(
                [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            model_path = os.path.join(checkpoint_dir, latest_checkpoint)
            model = load_local_model(model_path)
            print(f"Model loaded successfully from local checkpoint: {model_path}")
        else:
            raise FileNotFoundError("No local checkpoints found")
    
    # Make predictions
    probabilities = predict(model, example_features)
    
    # Print results
    print("\nCancer Risk Predictions:")
    for i, prob in enumerate(probabilities):
        print(f"\nSample {i+1}:")
        print(f"Cancer Probability: {prob[0]:.2%}")
        if prob[0] > 0.5:
            print("Risk Level: High")
        else:
            print("Risk Level: Low")
        print("Feature Values:")
        for feature_name, value in zip(streamlit_config.feature_names, example_features[i]):
            print(f"  {feature_name.replace('_', ' ').title()}: {value:.2f}")

if __name__ == "__main__":
    main() 