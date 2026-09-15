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

    def forward(self, X):
        x = self.layer1(X)
        x = self.relu(x)
        x = self.layer2(x)

        return x


class MoreLayersNN(nn.Module):
    def __init__(self, input, output, hidden_size1, hidden_size2):
        super().__init__()
        self.layer1 = nn.Linear(input, hidden_size1)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, output)

    def forward(self, X):
        x = self.layer1(X)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)

        return x


class ImprovedNN(nn.Module):
    def __init__(self, input, output, hidden_size1, hidden_size2):
        super().__init__() 
        self.layer1 = nn.Linear(input, hidden_size1)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, output)
        self.bnrm1 = nn.BatchNorm1d(hidden_size1)
        self.bnrm2 = nn.BatchNorm1d(hidden_size2)
        self.drop = nn.Dropout(0.5)

    def forward(self, X):
        x = self.layer1(X)
        x = self.bnrm1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.layer2(x)
        x = self.bnrm2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.layer3(x)

        return x
