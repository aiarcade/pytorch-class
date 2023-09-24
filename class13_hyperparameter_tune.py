import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the Iris dataset for demonstration purposes
data = load_iris()
X, y = data.data, data.target

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.int64)

# Define a PyTorch model with two nn.Linear layers
class Net(nn.Module):
    def __init__(self, trial):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], trial.suggest_int('n_units_l1', 16, 128))
        self.fc2 = nn.Linear(trial.suggest_int('n_units_l1', 16, 128), trial.suggest_int('n_units_l2', 16, 128))
        self.fc3 = nn.Linear(trial.suggest_int('n_units_l2', 16, 128), len(torch.unique(y_train)))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define an objective function to optimize
def objective(trial):
    model = Net(trial)
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_loguniform('lr', 1e-5, 1e-1))
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        predictions = model(X_valid)
        predicted_classes = torch.argmax(predictions, dim=1)
        accuracy = accuracy_score(y_valid, predicted_classes)

    return -accuracy  # Optuna minimizes, so we negate accuracy

# Create an Optuna study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)
