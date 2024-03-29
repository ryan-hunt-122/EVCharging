import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Critic(nn.Module):
    def __init__(self, shape):
        super(Critic, self).__init__()
        # Input: Grid (h x w), 1 channel (freq/action) X 2
        self.h = shape[0]
        self.w = shape[1]
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 4), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 4), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        # Fully connected layer 1
        self.num_in = (self.h - 2) * (self.w - 2) * 32
        self.fc1 = nn.Linear(in_features=1+self.num_in+sum(range(0, self.h+1)), out_features=200)
        # Fully connected layer 2
        self.fc2 = nn.Linear(in_features=200, out_features=200)
        # Output layer
        self.out = nn.Linear(in_features=200, out_features=1)
        nn.init.uniform_(self.out.weight.data, -0.0003, 0.0003)
        nn.init.uniform_(self.out.bias.data, -0.0003, 0.0003)

    def forward(self, x, s, u):
        # print(f'Critic Shape: {x.shape}')
        # Pass input through first convolutional layer
        x = F.relu(self.bn1(self.conv1(x)))
        # Pass output through second convolutional layer
        x = F.relu(self.bn2(self.conv2(x)))
        # Pass output through third convolutional layer
        x = F.relu(self.bn3(self.conv3(x)))
        # Flatten output from convolutional layers
        x = x.view(-1, self.num_in)
        # print(f'critic x: {x.shape}')
        # print(f'critic s: {s.shape}')
        # print(f'critic u: {u.shape}')
        x = torch.cat((x, s, u), 1)
        # Pass output through first fully connected layer
        x = F.relu(self.fc1(x))
        # Pass output through second fully connected layer
        x = F.relu(self.fc2(x))
        # Pass output through output layer
        x = self.out(x)
        # print(x)
        return x


