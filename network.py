import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetNeuralNetwork(nn.Module):
    def __init__(self):
        super(TargetNeuralNetwork, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv_2 = nn.Conv2d(64, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool_1 = nn.MaxPool2d(2)
        self.maxpool_2 = nn.MaxPool2d(2)
        self.maxpool_3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(320000, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        print(x.size())
        x = F.relu(self.conv_1(x))
        print(x.size())
        x = self.maxpool_1(x)
        print(x.size())
        x = F.relu(self.conv_2(x))
        print(x.size())
        x = self.maxpool_2(x)
        print(x.size())
        x = F.relu(self.conv_3(x))
        print(x.size())
        x = self.maxpool_3(x)
        print(x.size())
        x = x.view(-1, 320000)
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x_ = self.fc2(x)
        print(x_.size())
        return x_,x


class AttackNeuralNetwork(nn.Module):
    def __init__(self):
        super(AttackNeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(320000, 5000)
        self.fc2 = nn.Linear(5000, 128)      
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.relu(self.fc2(x))
        print(x.size())        
        x = self.fc3(x)
        print(x.size())
        return x