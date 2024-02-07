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
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(20, 1),
        )
    
    def forward(self, x):
        return self.approximator(x)

x = np.linspace(0, 2 * np.pi, 100)
# y = np.ones(x.shape)
y = np.sin(x)
x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

avg_losses = []
model = NN()
opt = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss(reduction="mean")

# layers = list(model.parameters())
# len(layers)
# layers[0].detach().numpy().reshape(-1,)
# layers[1].detach().numpy().reshape(-1,)
# layers[2].detach().numpy().reshape(-1,)
# layers[3].detach().numpy().reshape(-1,)

epochs = 200
for epoch in range(epochs):
    print(f"{epoch=}")
    predictions = []
    epoch_loss = 0
    model.train()
    for x0, y0 in zip(x, y):
        opt.zero_grad()
        output = model(x0)
        loss = criterion(output, y0)
        loss.backward()
        opt.step()

        predictions.append(output.item())
        epoch_loss += loss.item()

    avg_losses.append(epoch_loss / len(x))

plt.figure()
plt.plot(x, y, label='true', linewidth=2)
plt.plot(x, predictions, label='pred', linewidth=1)
plt.legend()
plt.show()


