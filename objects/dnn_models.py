import torch
from torch import nn
import numpy as np
import pandas as pd

from config import config


class SimpNN(nn.Module):
    def __init__(self, input, output, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(input, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output)
        # self.softmax = nn.Softmax()

    def forward(self, X):
        x = self.layer1(X)
        x = self.relu(x)
        x = self.layer2(x)
        # x = self.softmax(x)

        return x