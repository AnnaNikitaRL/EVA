import torch
from torch import nn
import torch.nn.functional as F

class Qnet(nn.Module):
    '''
    Deep convolutional network taking batch of stacks of frames as input
    (batch_size x in_channels x height x width)
    and returning vectors of size (n_actions, 1) and (embedding, 1).
    Current layer parameters correspond to img size = 84x84. 

    '''
    def __init__(self, n_actions, in_channels=4, embedding_dimension=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.fc = nn.Linear(in_features=7*7*32, out_features=embedding_dimension)
        self.output = nn.Linear(in_features=embedding_dimension, out_features=n_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        embed = self.fc(torch.flatten(x, start_dim=1))
        x = self.output(F.relu(embed))
        return x, embed

