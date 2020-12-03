import torch 
import torch.nn as nn

# Convolutional neural network (twohttps://duckduckgo.com/?q=install+gmsh+conda&t=canonical convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10, num_img_layers = 1, img_res = 128):
        super(ConvNet, self).__init__()
        self.fc0_1 = nn.Linear(img_res*img_res, int(img_res/2)*int(img_res/2))
        self.fc0_2 = nn.Linear(int(img_res/2)*int(img_res/2), int(img_res/2)*int(img_res/2))
        self.fc1 = nn.Linear(int(img_res/2)*int(img_res/2), 32*32)
        self.fc2 = nn.Linear(32*32, num_classes)
        
    def forward(self, x):
        #print(x.shape)
        out = x.reshape(x.size(0), -1)
        out = self.fc0_1(out)
        #print(out.shape)
        out = self.fc0_2(out)
        #print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        return out