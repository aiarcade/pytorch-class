import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
import torch.quantization as quantization

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Download and load the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define a simple ResNet18 model
model = resnet18(pretrained=False, num_classes=10).to(device)  # Move the model to GPU

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

import torch.quantization as quantization
# Enable quantization-aware training (QAT)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
quantized_model = quantization.prepare_qat(model)

# Train the quantization-aware model
num_epochs = 10
for epoch in range(num_epochs):
    quantized_model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Move data to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradient buffers
        optimizer.zero_grad()
        
        # Forward pass
        outputs = quantized_model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Convert the trained quantization-aware model for inference
quantized_model = quantization.convert(quantized_model)

# You can now use the quantized_model for inference on the GPU
