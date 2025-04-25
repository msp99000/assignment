import torch
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import model_config

def prepare_data(random_state=42):
    """Prepare synthetic cancer dataset"""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=model_config.n_samples,
        n_features=model_config.n_features,
        n_informative=model_config.n_informative,
        n_redundant=model_config.n_redundant,
        n_clusters_per_class=model_config.n_clusters_per_class,
        class_sep=model_config.class_sep,
        random_state=random_state
    )

    # Create DataFrame with medical feature names
    df = pd.DataFrame(X, columns=["tumor_size", "cell_count", "nuclei_density", "mitosis_rate"])

    # Preprocess
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=True
    )

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=model_config.batch_size
    )

    return train_loader, test_loader, scaler, df.columns.tolist() 