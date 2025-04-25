import requests
import numpy as np
import json

def make_prediction(features):
    # Prepare the request
    url = "http://localhost:8080/v1/models/cancer-classifier:predict"
    headers = {"Content-Type": "application/json"}
    
    # Format the input data
    data = {
        "instances": features.tolist()
    }
    
    # Make the prediction request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Prediction failed: {response.text}")

def main():
    # Example features (tumor_size, cell_count, nuclei_density, mitosis_rate)
    example_features = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Low risk
        [8.3, 7.9, 6.6, 5.8],  # High risk
        [6.9, 5.0, 4.2, 3.5]   # Medium risk
    ])
    
    try:
        # Make predictions
        predictions = make_prediction(example_features)
        
        # Print results
        print("\nCancer Risk Predictions:")
        for i, prob in enumerate(predictions['predictions']):
            print(f"\nSample {i+1}:")
            print(f"Cancer Probability: {prob[0]:.2%}")
            if prob[0] > 0.5:
                print("Risk Level: High")
            else:
                print("Risk Level: Low")
            print("Feature Values:")
            for feature_name, value in zip(["tumor_size", "cell_count", "nuclei_density", "mitosis_rate"], example_features[i]):
                print(f"  {feature_name.replace('_', ' ').title()}: {value:.2f}")
                
    except Exception as e:
        print(f"Error making predictions: {e}")

if __name__ == "__main__":
    main() 