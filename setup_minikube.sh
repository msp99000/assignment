#!/bin/bash

# Start Minikube if not already running
minikube status || minikube start

# Enable required addons
minikube addons enable metrics-server
minikube addons enable ingress

# Install KServe
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.10.0/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.10.0/serving-core.yaml
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.10.0/istio.yaml
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.10.0/net-istio.yaml
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml

# Wait for KServe to be ready
kubectl wait --for=condition=ready pod -l app=kserve-controller-manager -n kserve --timeout=300s

# Deploy MLflow server
kubectl apply -f mlflow-server.yaml

# Wait for MLflow server to be ready
kubectl wait --for=condition=ready pod -l app=mlflow-server --timeout=300s

# Forward MLflow server port
kubectl port-forward service/mlflow-server 5000:5000 &

# Train and register the model
python train.py

# Deploy the model to KServe
kubectl apply -f kserve.yaml

# Wait for the inference service to be ready
kubectl wait --for=condition=ready pod -l serving.kserve.io/inferenceservice=cancer-classifier --timeout=300s

# Get the inference service URL
echo "Inference Service URL:"
kubectl get inferenceservice cancer-classifier -o jsonpath='{.status.url}' 