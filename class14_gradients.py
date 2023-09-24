import torch

# Create tensors with requires_grad=True to enable autograd
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Define a simple computation
z = x * y

# Perform some more computations
w = z + 2

# Compute the gradients
w.backward()

# Print gradients
print("Gradient of w with respect to x:", x.grad)  # x.grad will be 3.0
print("Gradient of w with respect to y:", y.grad)  # y.grad will be 2.0
