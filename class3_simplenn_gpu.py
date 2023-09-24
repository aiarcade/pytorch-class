import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Step 1: Define the dataset
# Let's create a simple toy dataset with two features and binary labels.
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Step 2: Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Input layer with 2 features and 4 neurons in the hidden layer.
        self.fc2 = nn.Linear(4, 1)  # Output layer with 1 neuron for binary classification.

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Apply sigmoid activation to the hidden layer.
        x = torch.sigmoid(self.fc2(x))  # Apply sigmoid activation to the output layer.
        return x

# Step 3: Create an instance of the model and move to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)

# Step 4: Move the data to the GPU
X = X.to(device)
y = y.to(device)

# Step 5: Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification.
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent optimizer.
# Step 6: Training loop
num_epochs = 10000000000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# Step 7: Test the model
with torch.no_grad():
    test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
    predictions = model(test_inputs)
    predictions = (predictions > 0.5).float()  # Convert to binary (0 or 1) predictions
    print("Predictions:")
    print(predictions)

