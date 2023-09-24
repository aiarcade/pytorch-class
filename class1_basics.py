import torch
# Create tensors
tensor_a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
tensor_b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
# Basic tensor operations
tensor_sum = tensor_a + tensor_b
tensor_diff = tensor_a - tensor_b
tensor_product = tensor_a * tensor_b
tensor_division = tensor_a / tensor_b
# Print the results
print("Tensor A:", tensor_a)
print("Tensor B:", tensor_b)
print("Sum:", tensor_sum)
print("Difference:", tensor_diff)
print("Product:", tensor_product)
print("Division:", tensor_division)
# Calculate the gradient
tensor_sum.backward(torch.ones_like(tensor_sum)) 
# Print the gradients
print("Gradient of Tensor A:", tensor_a.grad)
print("Gradient of Tensor B:", tensor_b.grad)

