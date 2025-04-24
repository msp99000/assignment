import torch
import torch.nn as nn
from config import model_config

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(model_config.input_size, model_config.hidden_size)
        self.bn1 = nn.BatchNorm1d(model_config.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(model_config.hidden_size, model_config.output_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

    def predict(self, x):
        """Make predictions with the model"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

def create_model():
    """Create and return a new instance of IrisNet"""
    model = IrisNet()
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    model.apply(init_weights)
    return model 