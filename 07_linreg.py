import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1) # type: ignore

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples , n_features = x.shape
in_size = n_features
out_size = n_features
model = nn.Linear(in_size, out_size)

lr = 0.01
ls = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

iters = 100
for iter in range(1, iters, 1):
    y_pred = model(x)
    loss = ls(y_pred, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if iter % 10 == 0:
        w , b = model.parameters()
        print(f"iter = {iter}, w = {w[0][0]:.3f}, loss = {loss.item():.4f}")
    
predicted = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()
