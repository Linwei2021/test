import torch.nn as nn
import torch
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)

        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        #
        self.upconv4 = nn.ConvTranspose2d(in_channels = 128,out_channels = 64,kernel_size = 5,padding =2)


        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upconv6 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=5,
            padding = 2
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, padding=0),

        )

    # def crop(self,tensor,target_tensor):
    #     target_size = target_tensor.size()[2]
    #     tensor_size = tensor.size()[2]
    #     delta = tensor_size-target_size
    #     delta = delta//2
    #     return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]

    def forward(self, input):
        x1 = self.conv1(input)

        x2 = self.conv2(x1)#10,64,48,48

        x3 = self.conv3(x2) #10,128,40,40


        x4 = self.upconv4(x3)#10,64,44,44
        # y = self.crop(x2,x4)
        x4 = torch.cat([x4,x2],1)#10,128,44,44


        x5 = self.conv5(x4)

        x6 = self.upconv6(x5)
        # y = self.crop(x1, x6)
        x6 = torch.cat([x6, x1], 1)

        x7 = self.conv7(x6)

        x8 = self.conv8(x7)


        f = nn.Softmax(dim=1)
        result = f(x8)


        return result