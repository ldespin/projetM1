import torch.nn as nn
import torchvision.transforms as transforms
import sys
import torch
from numpy import genfromtxt

class Interpolate(nn.Module):
    def __init__(self, size):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size

    def forward(self, x):
        x = self.interp(x, size=self.size)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Upsampling
        self.down = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
            )
        # Upsampling
        self.up = nn.Sequential(
            nn.Linear(8*8*32,256),
            nn.ReLU(True),
            nn.Linear(256,6),
            nn.Softmax()
            )

    def forward(self, img):
        out = self.down(img)
        out = self.up(out.view(out.size(0), -1))
        return out