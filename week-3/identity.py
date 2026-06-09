import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


x = torch.linspace(-10, 10, 1000).unsqueeze(1)
y = x


class IdentityMLP(nn.Module):
    def __init__(self):
        super(IdentityMLP, self).__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)


model = IdentityMLP()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


epochs = 500

for epoch in range(epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")


W = model.layer.weight.item()
b = model.layer.bias.item()



print(f"\nFinal Loss: {loss.item():.6f}")
print(f"Learned Weight: {W:.6f}  (expected: 1.0)")
print(f"Learned Bias:   {b:.6f}  (expected: 0.0)")