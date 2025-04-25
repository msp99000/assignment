#!/bin/bash

# Start Minikube
minikube start --memory=4096 --cpus=2

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

# Create a local directory for model storage
mkdir -p models/cancer-classifier

# Copy the trained model to the local directory
cp -r mlruns/* models/cancer-classifier/

# Create a ConfigMap for the model
kubectl create configmap cancer-model --from-file=models/cancer-classifier -n kserve

# Apply the KServe configuration
kubectl apply -f kserve.yaml

# Wait for the inference service to be ready
kubectl wait --for=condition=ready pod -l serving.kserve.io/inferenceservice=cancer-classifier --timeout=300s

# Get the inference service URL
echo "Inference Service URL:"
kubectl get inferenceservice cancer-classifier -o jsonpath='{.status.url}'

# Forward the port
kubectl port-forward service/cancer-classifier-predictor-default 8080:80 