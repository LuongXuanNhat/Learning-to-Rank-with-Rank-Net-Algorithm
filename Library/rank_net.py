import torch
import torch.nn as nn
import torch.optim as optim

class RankNet(nn.Module):
    def __init__(self, input_size):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)
