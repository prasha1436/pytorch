import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# device config
device = torch.device('cpu')

# hyper paramters
#input_size = 784 # 28*28 = 784
#hidden_size = 100
#num_classes = 10
num_epochs = 4
batch_size = 4
lr = 0.001

#CIFAR 10 Dataset
transform = transform.Compose([transform.ToTensor(),transform.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5))])

train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform, download = True)

test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = transform)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


""" for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap= 'gray')
plt.show() """

# implement Conv NN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 140)
        self.fc2 = nn.Linear(140, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x.view(-1, 16*5*5)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = ConvNet()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

# training loop
n_total = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28 shape
        # input = 784
        # num_batches, input_size
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'epoch = {epoch+1}/{num_epochs} , step {i+1}/{n_total}, loss = {loss.item():.4f}')

print('Finished Training')

# testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # torch.max gives value, index (class/prediction)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct/n_samples
    print(f'Accuracy of the network= {acc}%')

    for i in range(10):
        acc = 100.0 * n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy of the {classes[i]}= {acc}%')


