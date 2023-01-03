import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from torch.optim import Adam
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

N = 1_000_000
D = 3
bound = 10.

batch = N//1000
epochs = 30
hidden_dimension = 8
learning_rate = 1e-3

force_train = False

def generate_dataset(N, D, a, b):
    unif = Uniform(a, b)
    U = unif.sample((N, D))
    G = torch.normal(U, torch.ones_like(U))
    return U, G

class Estimator(nn.Module):
    def __init__(self, D, h) -> None:
        super().__init__()
        self.fw = nn.Sequential(
            nn.Linear(D, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, D)
        )
        return

    def forward(self, x):
        return self.fw(x)

label, data = generate_dataset(N, D, -bound, bound)
label, data = (label.to(device=device), data.to(device=device))
model = Estimator(D, hidden_dimension).to(device=device)
loss = nn.MSELoss()
L = []
optimizer = Adam(model.parameters(), lr = learning_rate)

if not os.path.isfile("model.pt") or force_train:
    for _ in range(epochs):
        for i in range(N//batch):
            optimizer.zero_grad()
            model_input = data[i*batch : (i+1)*batch]
            expected_output = label[i*batch : (i+1)*batch]
            model_output = model(model_input)
            l = loss(model_output, expected_output)
            l.backward()
            optimizer.step()
            L.append(l.item())
    L = torch.tensor(L)
    torch.save(model.state_dict(), "model.pt")
    torch.save(L, "loss.pt")
else:
    model.load_state_dict(torch.load("model.pt"))
    L = torch.load("loss.pt")


print(f"last few losses:\n {L[-10:]}")
plt.loglog(L)
plt.show()


print(model(torch.tensor([[1., 1., 1.], [0., 0., 0.], [12., 12., 12.]]).to(device=device)))
