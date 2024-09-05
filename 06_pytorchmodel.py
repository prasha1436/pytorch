import torch
import torch.nn as nn

x = torch.tensor([[1],[2],[3]], dtype = torch.float32)
y = torch.tensor([[2],[4],[6]], dtype = torch.float32)

x_test = torch.tensor([5], dtype = torch.float32)
n_samples, n_features = x.shape
in_size = n_features
out_size = n_features
model = nn.Linear(in_size, out_size)

print(f"Prediction before training: f(5) = {model(x_test).item():.3f}")

lr = 0.01
iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

for i in range(1, iters, 1):
	y_pred = model(x)

	ls = loss(y, y_pred)

	ls.backward()

	optimizer.step()

	optimizer.zero_grad()

	if i %10 == 0:
		w, b = model.parameters()
		print(f" iter = {i} w = {w[0][0]:.3f} loss = {ls:.8f}")

print(f"Prediction after training: f(5) = {model(x_test).item():.3f}")