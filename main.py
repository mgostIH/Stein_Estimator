import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from torch.optim import Adam
import matplotlib.pyplot as plt
import os
import gc

N = 1_000_000
D = 3
bound = 10.

batch = N//20
epochs = 1000
hidden_dimension = 2
learning_rate = 1e-3

force_train = True

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
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, D)
        )
        return

    def forward(self, x):
        return self.fw(x)

label, data = generate_dataset(N, D, -bound, bound)
model = Estimator(D, hidden_dimension)
loss = nn.MSELoss()
#L = []
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
            #L.append(l.detach().reshape((1,)))
    #L = torch.cat(L).detach().numpy()
    torch.save(model.state_dict(), "model.pt")
    #torch.save(L, "loss.pt")
else:
    model.load_state_dict(torch.load("model.pt"))
    #L = torch.load("loss.pt")


#print(f"last few losses:\n {L[-10:]}")
#plt.loglog(L)
#plt.show()


print(model(torch.tensor([[1., 1., 1.], [0., 0., 0.], [12., 12., 12.]])))
