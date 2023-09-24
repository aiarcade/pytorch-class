import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

# Step 1: Create a Custom Dataset with Features and Labels
class CustomDataset(Dataset):
    def __init__(self, features_list, labels_list, transform=None):
        self.features = features_list
        self.labels = labels_list
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Step 2: Instantiate the Custom Dataset with Data Augmentation
features_list = [1.0, 2.0, 3.0, 4.0, 5.0]
labels_list = [10.0, 20.0, 30.0, 40.0, 50.0]

# Define a custom transform for data augmentation (e.g., random scaling)
def custom_transform(x):
    scale_factor = random.uniform(0.8, 1.2)
    return x * scale_factor

transform = transforms.Compose([transforms.Lambda(custom_transform)])

custom_dataset = CustomDataset(features_list, labels_list, transform=transform)

# Step 3: Create a Data Loader
batch_size = 2
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Step 4: Iterate Over Batches
for batch in data_loader:
    features, labels = batch
    print("Features:", features)
    print("Labels:", labels)
