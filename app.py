import streamlit as st
import torch
import numpy as np
from model import create_model
from config import streamlit_config, model_config
import mlflow
from data import prepare_data

def load_model():
    """Load the latest model from MLflow"""
    mlflow.set_tracking_uri("http://localhost:5000")
    model = mlflow.pytorch.load_model("models:/cancer-classifier/latest")
    model.eval()
    return model

def predict(model, features):
    """Make prediction using the model"""
    with torch.no_grad():
        features = torch.tensor(features, dtype=torch.float32)
        prediction = model(features)
        return prediction.numpy()[0]

def main():
    st.title(streamlit_config.title)
    st.write(streamlit_config.description)

    # Load model
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

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
        # Prepare input
        input_features = np.array([[features[f] for f in streamlit_config.feature_names]])
        
        # Make prediction
        prediction = predict(model, input_features)
        
        # Display results
        st.header("Prediction Results")
        st.write(f"Probability of Cancer: {prediction:.2%}")
        
        # Add interpretation
        if prediction > 0.5:
            st.error("High probability of cancer detected")
        else:
            st.success("Low probability of cancer detected")

    # Add model information
    st.sidebar.header("Model Information")
    st.sidebar.write("Model Architecture:")
    st.sidebar.write(f"- Input Size: {model_config.input_size}")
    st.sidebar.write(f"- Hidden Size: {model_config.hidden_size}")
    st.sidebar.write(f"- Output Size: {model_config.output_size}")
    
    # Add feature importance visualization
    st.sidebar.header("Feature Importance")
    feature_importance = {
        "tumor_size": 0.3,
        "cell_count": 0.25,
        "nuclei_density": 0.25,
        "mitosis_rate": 0.2
    }
    st.sidebar.bar_chart(feature_importance)

if __name__ == "__main__":
    main()
