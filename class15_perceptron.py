import torch
import torch.nn as nn
import torch.optim as optim

# Create a single perceptron with two input features
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.weight = nn.Parameter(torch.randn(2))  # Learnable weights
        self.bias = nn.Parameter(torch.randn(1))    # Learnable bias

    def forward(self, x):
        # Compute the weighted sum of inputs and add the bias
        weighted_sum = torch.sum(self.weight * x) + self.bias
        # Apply an activation function (e.g., sigmoid)
        output = torch.sigmoid(weighted_sum)
        return output

# Create an instance of the perceptron
perceptron = Perceptron()

# Sample dataset (two-dimensional input and binary output)
X = torch.tensor([[1.0, -1.0], [2.0, 1.0], [-1.0, -3.0], [-2.0, 2.0]])
y = torch.tensor([1, 1, 0, 0], dtype=torch.float32)  # Keep y as 1D tensor

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(perceptron.parameters(), lr=0.1)

# Training loop

for i in range(0,len(X)):
    # Forward pass
    outputs = perceptron(X[i])
    # Compute the loss
    loss = criterion(outputs[0], y[i])
    
    # Backpropagation: Compute gradients
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient descent: Update weights and bias
    optimizer.step()


# Test the trained perceptron
test_data = torch.tensor([[0.5, 0.5]])
with torch.no_grad():
    predictions = perceptron(test_data)
    print("Predictions:", predictions)
