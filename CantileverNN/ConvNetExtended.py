'''
@Title: ConvNet Class
@Description: Defines neural network architecture for ConvNet
@Author: Philippe Wyder (pmw2125@columbia.edu)
'''

import torch 
import torch.nn as nn

# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes=10, num_img_layers = 1, img_res = 128):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,  
            # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.Conv2d(num_img_layers, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(32*int(img_res/4)*int(img_res/4), 32*32)
        self.fc2 = nn.Linear(32*32, num_classes)
        
    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
