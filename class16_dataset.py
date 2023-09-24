import torch
from torch.utils.data import Dataset

class NumbersDataset(Dataset):
    def __init__(self, features_list, labels_list):
        self.features = features_list
        self.labels = labels_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Example lists of features and labels
features_list = [1.0, 2.0, 3.0, 4.0, 5.0]
labels_list = [10.0, 20.0, 30.0, 40.0, 50.0]

# Create an instance of the custom dataset
custom_dataset = NumbersDataset(features_list, labels_list)

# Example: Accessing the first element
features, labels = custom_dataset[0]
print("Feature:", features)
print("Label:", labels)
