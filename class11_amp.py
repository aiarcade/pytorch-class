import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from apex import amp  # Apex is a library for mixed-precision training

# Define a custom dataset (you can replace this with your actual dataset loading)
class RandomDataset(Dataset):
    def __init__(self, size=1000, num_classes=10):
        self.data = torch.rand(size, 784)  # Random data of shape (size, 784)
        self.labels = torch.randint(0, num_classes, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create an instance of the model and move it to the GPU
model = SimpleNet().cuda()

# Create an instance of the custom dataset
random_dataset = RandomDataset(size=1000, num_classes=10)

# Create a data loader for the dataset
dataloader = DataLoader(random_dataset, batch_size=64, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Enable mixed-precision training with Apex
model, optimizer = amp.initialize(model, optimizer, opt_level="O2")  # "O2" enables mixed-precision training

# Training loop
for epoch in range(5):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()  # Backpropagate using mixed precision
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{5}, Loss: {running_loss / len(dataloader)}")

# Save or evaluate the trained model as needed
