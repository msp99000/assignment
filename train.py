import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch
import numpy as np
import os
from model import create_model
from data import prepare_data
from config import model_config, mlflow_config

def save_checkpoint(model, optimizer, epoch, loss, path='checkpoints'):
    """Save model checkpoint locally"""
    if not os.path.exists(path):
        os.makedirs(path)
    
    checkpoint_path = os.path.join(path, f'model_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def evaluate_model(model, test_loader, loss_fn):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def train_model():
    # Set up MLflow
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.set_experiment(mlflow_config.experiment_name)

    # Prepare data
    train_loader, test_loader, scaler = prepare_data()

    # Create model and optimizer
    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=model_config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "input_size": model_config.input_size,
            "hidden_size": model_config.hidden_size,
            "output_size": model_config.output_size,
            "learning_rate": model_config.learning_rate,
            "batch_size": model_config.batch_size,
            "epochs": model_config.epochs,
            "optimizer": "Adam",
            "loss_fn": "CrossEntropyLoss"
        })

        # Training loop
        best_test_accuracy = 0
        for epoch in range(model_config.epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            # Calculate metrics
            train_accuracy = 100 * correct / total
            avg_loss = total_loss / len(train_loader)
            
            # Evaluate on test set
            test_loss, test_accuracy = evaluate_model(model, test_loader, loss_fn)

            # Log metrics
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy
            }, step=epoch)

            print(f"Epoch {epoch+1}/{model_config.epochs}")
            print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

            # Save checkpoint locally
            checkpoint_path = save_checkpoint(model, optimizer, epoch, avg_loss)
            print(f"Saved checkpoint to {checkpoint_path}")

            # Save best model to MLflow
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                mlflow.pytorch.log_model(
                    model,
                    artifact_path=mlflow_config.model_name,
                    registered_model_name=mlflow_config.model_name
                )
                print(f"New best model saved with test accuracy: {test_accuracy:.2f}%")

        print(f"Training completed. Best test accuracy: {best_test_accuracy:.2f}%")
        print(f"Model logged to MLflow with run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train_model() 