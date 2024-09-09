import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self, transform= None):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype= np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs , targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransorm:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs , targets = sample
        inputs *= self.factor
        return inputs, targets


dataset = WineDataset(transform= ToTensor())
first = dataset[0]
fea , lab = first
print(type(fea), type(lab))
print(fea)

composed = torchvision.transforms.Compose([ToTensor(), MulTransorm(2)])
dataset = WineDataset(transform=composed)
first = dataset[0]
fea , lab = first
print(type(fea), type(lab))
print(fea)