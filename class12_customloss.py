import torch
import torch.nn as nn
import torch.optim as optim

# Define a custom loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted, target):
        # Calculate the Mean Squared Error (MSE) loss
        mse_loss = torch.mean((predicted - target) ** 2)
        return mse_loss

# Create a simple neural network with nn.Linear
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Generate some example data
input_size = 1
output_size = 1
input_data = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float32)
target_data = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]], dtype=torch.float32)

# Define a custom loss function
loss_function = CustomLoss()

# Instantiate the neural network
model = SimpleNN(input_size, output_size)

# Define an optimizer (e.g., SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predicted_output = model(input_data)
    
    # Calculate the custom loss
    loss = loss_function(predicted_output, target_data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    test_input = torch.tensor([[6.0]], dtype=torch.float32)
    predicted_output = model(test_input)
    print(f'Test Input: {test_input.item()}, Predicted Output: {predicted_output.item()}')
