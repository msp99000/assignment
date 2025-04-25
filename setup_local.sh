#!/bin/bash

# Clean up existing resources
kubectl delete -f kserve.yaml --ignore-not-found
kubectl delete pvc cancer-model -n kserve --ignore-not-found
kubectl delete pv cancer-model-pv --ignore-not-found

# Start Minikube
minikube start --memory=2048 --cpus=2

# Enable required addons
minikube addons enable metrics-server
minikube addons enable ingress

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.3/cert-manager.yaml

# Wait for cert-manager to be ready
echo "Waiting for cert-manager to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=300s

# Install Knative Serving CRDs
echo "Installing Knative Serving CRDs..."
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.10.0/serving-crds.yaml

# Install Knative Serving core components
echo "Installing Knative Serving core components..."
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.10.0/serving-core.yaml

# Install Istio
echo "Installing Istio..."
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.10.0/istio.yaml
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.10.0/net-istio.yaml

# Wait for Knative to be ready
echo "Waiting for Knative to be ready..."
kubectl wait --for=condition=ready pod -l app=networking-istio -n knative-serving --timeout=300s

# Install KServe CRDs
echo "Installing KServe CRDs..."
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve-crds.yaml

# Install KServe
echo "Installing KServe..."
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml

# Create namespace if it doesn't exist
kubectl create namespace kserve --dry-run=client -o yaml | kubectl apply -f -

# Wait for KServe to be ready
echo "Waiting for KServe to be ready..."
kubectl wait --for=condition=ready pod -l app=kserve-controller-manager -n kserve --timeout=300s

# Create a local directory for model storage
mkdir -p models/cancer-classifier

# Copy the trained model to the local directory
cp -r mlruns/* models/cancer-classifier/

# Create a PersistentVolume for model storage
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: cancer-model-pv
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "$(pwd)/models/cancer-classifier"
EOF

# Create a PersistentVolumeClaim
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: cancer-model
  namespace: kserve
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
EOF

# Wait for PV and PVC to be ready
echo "Waiting for storage to be ready..."
kubectl wait --for=condition=ready pv cancer-model-pv --timeout=300s
kubectl wait --for=condition=ready pvc cancer-model -n kserve --timeout=300s

# Apply the KServe configuration
echo "Applying KServe configuration..."
kubectl apply -f kserve.yaml

# Wait for the inference service to be ready
echo "Waiting for inference service to be ready..."
kubectl wait --for=condition=ready pod -l serving.kserve.io/inferenceservice=cancer-classifier -n kserve --timeout=300s

# Get the inference service URL
echo "Inference Service URL:"
kubectl get inferenceservice cancer-classifier -n kserve -o jsonpath='{.status.url}'

# Forward the port
echo "Forwarding port 8080..."
kubectl port-forward service/cancer-classifier-predictor-default -n kserve 8080:80 