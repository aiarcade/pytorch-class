import torch
import torch.nn as nn
import torch.optim as optim

class LargeModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LargeModel, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

def train_large_model():
    # Define hyperparameters
    input_size = 4096
    hidden_size = 4096
    num_layers = 200
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 1000

    # Create a LargeModel instance
    model = LargeModel(input_size, hidden_size, num_layers, num_classes).cuda()

    data = torch.randn(100, input_size).cuda()
    labels = torch.randint(0, num_classes, (100,)).cuda()

    # Loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

if __name__ == "__main__":
    train_large_model()
