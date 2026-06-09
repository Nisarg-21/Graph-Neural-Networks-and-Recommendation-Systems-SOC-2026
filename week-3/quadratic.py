import torch
import torch.nn as nn
import torch.optim as optim

x = torch.linspace(-10, 10, 1000).unsqueeze(1)
y = x ** 2

class SquareMLP(nn.Module):
    def __init__(self):
        super(SquareMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = SquareMLP()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 5000
for epoch in range(epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

print(f"\nFinal Loss: {loss.item():.6f}")