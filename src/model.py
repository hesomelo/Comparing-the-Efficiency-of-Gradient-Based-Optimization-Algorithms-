import torch as t
import torch.optim
import torch.nn as nn
from src.optimizers import Sgd



inp = 1
hid = 10
out = inp

model = nn.Sequential(
    nn.Linear(inp,hid),
    nn.ReLU(),
    nn.Linear(hid,hid),
    nn.ReLU(),
    nn.Linear(hid,out),
)



optimizer = Sgd(model.parameters(), lr=0.001)