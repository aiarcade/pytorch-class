
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []

        if self.train:
            for i in range(1, 6):
                batch_path = os.path.join(root, f'data_batch_{i}')
                data, targets = self.load_batch(batch_path)
                self.data.extend(data)
                self.targets.extend(targets)
        else:
            test_batch_path = os.path.join(root, 'test_batch')
            self.data, self.targets = self.load_batch(test_batch_path)
        
    def load_batch(self, batch_path):
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        
        data = batch['data'].reshape(-1, 3, 32, 32)
        targets = batch['labels']
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = img.transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

        if self.transform:
            img = self.transform(img)

        return img, target



