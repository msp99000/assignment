import torch
import mlflow
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from config import mlflow_config
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
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    
    return predicted_class.numpy(), probabilities.numpy()

def main():
    # Example features (sepal length, sepal width, petal length, petal width)
    example_features = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.3, 2.9, 5.6, 1.8],  # Virginica
        [5.9, 3.0, 4.2, 1.5]   # Versicolor
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
    predictions, probabilities = predict(model, example_features)
    
    # Print results
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    print("\nPredictions:")
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        print(f"\nSample {i+1}:")
        print(f"Predicted class: {class_names[pred]}")
        print("Class probabilities:")
        for class_name, prob in zip(class_names, probs):
            print(f"  {class_name}: {prob:.4f}")

if __name__ == "__main__":
    main() 