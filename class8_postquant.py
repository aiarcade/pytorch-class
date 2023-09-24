import torch
import torchvision.models as models
import torch.quantization as quantization
import time
from torchsummary import summary

# Load a pre-trained model (e.g., ResNet18)
model = models.resnet18(pretrained=True)

# Create a dummy input with the same shape as expected by the model
dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image

# Prepare the model for quantization
quantized_model = quantization.quantize_dynamic(
    model,  # Original pre-trained model
    {torch.nn.Conv2d, torch.nn.Linear},  # Specify which layers to quantize
    dtype=torch.qint8  # Specify the quantization data type (int8)
)

# Set the model to evaluation mode
quantized_model.eval()

# Warm up the model
for _ in range(10):
    quantized_model(dummy_input)

# Measure inference time for the original model
start_time = time.time()
for _ in range(100):
    model(dummy_input)
end_time = time.time()
original_model_time = (end_time - start_time) / 100

# Measure inference time for the quantized model
start_time = time.time()
for _ in range(100):
    quantized_model(dummy_input)
end_time = time.time()
quantized_model_time = (end_time - start_time) / 100

# Calculate speedup factor
speedup_factor = original_model_time / quantized_model_time

print(f"Inference time for original model: {original_model_time:.4f} seconds")
print(f"Inference time for quantized model: {quantized_model_time:.4f} seconds")
print(f"Speedup factor: {speedup_factor:.2f}x")

summary(model, input_size=(3, 224, 224), device='cpu')
summary(quantized_model, input_size=(3, 224, 224), device='cpu')
