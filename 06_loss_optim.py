import torch
import torch.nn as nn

x = torch.tensor([1,2,3], dtype = torch.float32)
y = torch.tensor([2,4,6], dtype = torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)

def forward(x):
	return w*x


print(f"Prediction before training: f(5) = {forward(5):.3f}")

lr = 0.01
iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr = lr)

for i in range(1, iters, 1):
	y_pred = forward(x)

	ls = loss(y, y_pred)

	ls.backward()

	optimizer.step()

	optimizer.zero_grad()

	if i %10 == 0:
		print(f" iter = {i} w = {w:.3f} loss = {ls:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")