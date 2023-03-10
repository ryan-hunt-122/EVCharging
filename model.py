import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DFQI(nn.Module):
    def __init__(self):
        super(DFQI, self).__init__()
        self.dimension = 10
        self.conv1 = nn.Conv2d(self.dimension, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(9216, 500)
        self.fc2 = nn.Linear(500, self.dimension)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Agent():
    def __init__(self, device):
        self.device = device
        self.policy_net = DFQI().to(self.device)
        # self.target_net = DFQI().to(self.device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.00025, momentum=0.95, alpha=0.01)

    def forward(self, x):
        return self.policy_net.forward(x)

    def update(self, L_b):
        loss = L_b

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
