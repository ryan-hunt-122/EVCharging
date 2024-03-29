import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, shape):
        super(Actor, self).__init__()
        # Input: Grid (h x w), 1 channel (freq/action)
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
        # Shape is given by Y x Y x C -> Y = [W-K+2]+1
        self.fc1 = nn.Linear(in_features=3 + ((self.h - 2) * (self.w - 2) * 32), out_features=200)
        # Fully connected layer 2
        self.fc2 = nn.Linear(in_features=200, out_features=200)
        # Output layer
        self.out = nn.Linear(in_features=200, out_features=1)
        nn.init.uniform_(self.out.weight.data, -0.0003, 0.0003)
        nn.init.uniform_(self.out.bias.data, -0.0003, 0.0003)

    def forward(self, x, s, h, w):
        # print(f'Actor shape: {x.shape}')
        # Pass input through first convolutional layer
        x = self.bn1(self.conv1(x))
        # Pass output through second convolutional layer
        x = F.relu(self.bn2(self.conv2(x)))
        # Pass output through third convolutional layer
        x = F.relu(self.bn3(self.conv3(x)))
        # Flatten output from convolutional layers
        x = x.view(x.size(0), -1)
        # print(f'x shape: {x.shape}, s shape: {s.shape}')
        # print(f'h shape: {h.shape}, w shape: {w.shape}')
        x = torch.cat((x, s, h, w), 1)
        # Pass output through first fully connected layer
        x = F.relu(self.fc1(x))
        # Pass output through second fully connected layer
        x = F.relu(self.fc2(x))
        # Pass output through output layer
        x = torch.sigmoid(self.out(x))
        return x
