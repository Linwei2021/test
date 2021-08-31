import torch.nn as nn
import torch
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        
        )

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #
        # )
        
        
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, padding=0),
            
        )

    def addInput(self,input,x):
        input = input.numpy()
        x = x.detach().numpy()
        input = input.transpose(1,0,2,3)
        x = np.insert(x,1,input,1)
        result = torch.from_numpy(x)
        return result

    def forward(self, input):
        x = self.conv1(input)
        # x = self.addInput(input, x)  # 10,33,64,64

        x = self.conv2(x)
        # x = self.addInput(input, x)  # 10,65,64,64

        x = self.conv3(x)
        # x = self.addInput(input, x)#10,129,64,64

        x = self.conv4(x)

        # x = self.conv5(x)

        f = nn.Softmax(dim=1)
        x = f(x)

        return x
        