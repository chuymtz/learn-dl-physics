import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class NN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.approximator = nn.Sequential(
            nn.Linear(1, 150),
            nn.ReLU(),
            nn.Linear(150, 1),
        )
    
    def forward(self, x):
        return self.approximator(x)

radians = np.linspace(0, 2 * np.pi, 100)
sine_values = np.sin(radians)
radians = torch.tensor(radians, dtype=torch.float32).view(-1, 1)
sine_values = torch.tensor(sine_values, dtype=torch.float32).view(-1, 1)

avg_losses = []
model = NN()
opt = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss(reduction="mean")


epochs = 200
for epoch in range(epochs):
    print(f"{epoch=}")
    predictions = []
    epoch_loss = 0
    model.train()
    for radian, target in zip(radians, sine_values):
        opt.zero_grad()
        output = model(radian)
        loss = criterion(output, target)
        loss.backward()
        opt.step()

        predictions.append(output.item())
        epoch_loss += loss.item()

    avg_losses.append(epoch_loss / len(radians))

plt.figure()
plt.plot(radians, sine_values, label='true', linewidth=2)
plt.plot(radians, predictions, label='pred', linewidth=1)
plt.legend()
plt.show()



