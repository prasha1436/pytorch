import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler as sc
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target # type: ignore

n_samples , n_features = x.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

sc = sc()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))    
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# model
class LogReg(nn.Module):
    def __init__(self, n_in_features):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(n_in_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogReg(n_features)

lr = 0.01
ls = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= lr)

iters = 100
for iter in range(1, iters+1, 1):
    y_pred = model(x_train)
    loss = ls(y_pred, y_train)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if iter % 10 == 0:
        print(f"iter = {iter}, loss = {loss.item():.4f}")

with torch.no_grad():
    y_pred = model(x_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f"Accuracy= {acc:.4f}")
