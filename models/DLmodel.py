import torch.nn as nn
from skimage import img_as_float
import torch
import torch.nn.functional as F

class Net(nn.Module):
        def __init__(self, num_of_classes):
            super(Net, self).__init__()
            # input image channel, output channels, kernel size square convolution
            # kernel
            # input size = 102, output size = 100
            self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            # input size = 50, output size = 48
            self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            # input size = 24, output size = 24
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
            self.vp = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(256*12*12, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, num_of_classes)
        
        def forward(self, x):
            in_size = x.size(0)
            x = F.relu(self.bn1(self.vp(self.conv1(x))))
            x = F.relu(self.bn2(self.vp(self.conv2(x))))
            x = F.relu(self.bn3(self.vp(self.conv3(x))))
            x = self.drop2D(x)
            x = x.view(in_size, -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x