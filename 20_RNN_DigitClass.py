import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# device config
device = torch.device('cpu')

# hyper paramters
#input_size = 784 # 28*28 = 784
input_size = 28
sequence_length = 28
hidden_size = 128
num_classes = 10
num_epochs = 2
batch_size = 100
lr = 0.001
num_layers = 2

#MNIST

transform = transform.Compose([transform.ToTensor(),
                               transform.Normalize((0.1307, ), (0.3081, ))])

train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transform, download = True)

test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transform)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

examples = iter(train_loader)
samples, lables = next(examples)
print(samples.shape, lables.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap= 'gray')
plt.show()

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out , _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# training loop
n_total = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28 shape
        # input = 784
        # num_batches, input_size
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch = {epoch+1}/{num_epochs} , step {i+1}/{n_total}, loss = {loss.item():.4f}')

# testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # torch.max gives value, index (class/prediction)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct/n_samples
    print('Accuracy = ', acc)


torch.save(model.state_dict(),"mnist_ffn.pth")