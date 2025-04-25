import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Set up MLflow tracking
mlflow.set_experiment("Cancer Detection Experiments")

# Define a simple PyTorch classification model
class CancerClassifier(nn.Module):
    def __init__(self, input_size):
        super(CancerClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Function to train and log an experiment
def run_experiment(X, y, run_name, lr=0.001, epochs=50, batch_size=16):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                   torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = CancerClassifier(input_size=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        # Training loop
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            preds = model(test_tensor).round()
            acc = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds)
            recall = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.pytorch.log_model(model, "model")

        print(f"{run_name} -> Acc: {acc:.2f}, Precision: {precision:.2f}, F1: {f1:.2f}")

# --- Generate and Run 5 Datasets ---
for i in range(5):
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42 + i
    )

    # Replace column names with medical ones
    df = pd.DataFrame(X, columns=["tumor_size", "cell_count", "nuclei_density", "mitosis_rate"]) #type: ignore
    run_experiment(df.values, y, run_name=f"Cancer Run {i+1}")
