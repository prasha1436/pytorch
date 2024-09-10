import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
from torch.utils.tensorboard.writer import SummaryWriter
writer = SummaryWriter("runs/MNIST2")
# device config
device = torch.device('cpu')

# hyper paramters
input_size = 784 # 28*28 = 784
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
lr = 0.01

#MNIST

train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transform.ToTensor(), download = True)

test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transform.ToTensor())

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

examples = iter(train_loader)
samples, lables = next(examples)
#sys.exit()

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap= 'gray')
#plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)
writer.close()
#sys.exit()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out) 
        out = self.l2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

writer.add_graph(model, samples.reshape(-1, 28*28))
writer.close()
#sys.exit()

# training loop
n_total = len(train_loader)
running_loss = 0.0 
running_correct = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28 shape
        # input = 784
        # num_batches, input_size
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs, 1)
        running_loss += loss.item()
        running_correct += (predictions == labels).sum().item()

        if (i+1) % 100 == 0:
            print(f'epoch = {epoch+1}/{num_epochs} , step {i+1}/{n_total}, loss = {loss.item():.4f}')
            writer.add_scalar('Training Loss', running_loss/100, epoch*n_total+i)
            writer.add_scalar('Accurary', running_correct/100, epoch*n_total+i)
            running_loss = 0.0
            running_correct = 0

# testing
label = []
preds = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    i = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        # torch.max gives value, index (class/prediction)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        
        class_preds = [F.softmax(output, dim=0) for output in outputs]
        preds.append(class_preds)
        label.append(labels)

    label = torch.cat(label)
    preds = torch.cat([torch.stack(batch) for batch in preds])

    acc = 100.0 * n_correct/n_samples
    print('Accuracy = ', acc)

    classes = range(10)
    for i in range(10):
        labels_i = label == i
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    
    writer.close


