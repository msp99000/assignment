import requests
import numpy as np
import json

def make_prediction(features):
    """Make prediction using KServe endpoint"""
    url = "http://localhost:8080/v2/models/cancer-classifier/infer"
    headers = {"Content-Type": "application/json"}
    
    # Format input according to V2 protocol
    data = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [1, 4],
                "datatype": "FP32",
                "data": features.tolist()
            }
        ]
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Prediction failed: {response.text}")

def main():
    # Example features for different risk levels
    low_risk = np.array([[0.1, 0.2, 0.1, 0.1]])  # Low risk features
    medium_risk = np.array([[0.5, 0.5, 0.5, 0.5]])  # Medium risk features
    high_risk = np.array([[0.9, 0.9, 0.9, 0.9]])  # High risk features
    
    # Make predictions
    print("Making predictions...")
    print("\nLow risk sample:")
    result = make_prediction(low_risk)
    print(f"Probability: {result['outputs'][0]['data'][0]:.2%}")
    
    print("\nMedium risk sample:")
    result = make_prediction(medium_risk)
    print(f"Probability: {result['outputs'][0]['data'][0]:.2%}")
    
    print("\nHigh risk sample:")
    result = make_prediction(high_risk)
    print(f"Probability: {result['outputs'][0]['data'][0]:.2%}")

if __name__ == "__main__":
    main() 