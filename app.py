import streamlit as st
import requests
import numpy as np
import json
from config import streamlit_config

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
    st.title(streamlit_config.title)
    st.write(streamlit_config.description)

    # Create input form
    st.header("Input Features")
    col1, col2 = st.columns(2)
    
    features = {}
    for i, feature in enumerate(streamlit_config.feature_names):
        with col1 if i % 2 == 0 else col2:
            min_val, max_val = streamlit_config.feature_ranges[feature]
            features[feature] = st.slider(
                f"{feature.replace('_', ' ').title()}",
                min_val,
                max_val,
                (min_val + max_val) / 2
            )

    # Make prediction
    if st.button("Predict"):
        try:
            # Prepare input
            input_features = np.array([[features[f] for f in streamlit_config.feature_names]])
            
            # Make prediction
            result = make_prediction(input_features)
            
            # Extract probability from V2 response
            probability = result['outputs'][0]['data'][0]
            
            # Display results
            st.header("Prediction Results")
            st.write(f"Cancer Probability: {probability:.2%}")
            
            # Add interpretation
            if probability > 0.5:
                st.error("High probability of cancer detected")
            else:
                st.success("Low probability of cancer detected")
                
            # Display feature importance
            st.sidebar.header("Feature Importance")
            feature_importance = {
                "tumor_size": 0.3,
                "cell_count": 0.25,
                "nuclei_density": 0.25,
                "mitosis_rate": 0.2
            }
            st.sidebar.bar_chart(feature_importance)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Make sure the KServe service is running and accessible at localhost:8080")

if __name__ == "__main__":
    main()
