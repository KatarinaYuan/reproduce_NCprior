import torch
from torch import nn


class SELayer(nn.Module):

    def __init__(self, in_channels, reduction=1):

        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bsz, c, _, _ = x.size()
        y = self.avg_pool(x).view(bsz, c)
        y = self.fc(y).view(bsz, c, 1, 1)
        return x * y.expand_as(x)



class ResidualBlock(nn.Module):

    def __init__(self,):
        super(ResidualBlock, self).__init__()

    def _construct_conv(self, hidden_dims, kernel_sizes, strides, paddings):
        L = len(hidden_dims) - 1
        modules = []

        for i in range(L):
            modules.append(nn.Sequential(
                    nn.BatchNorm2d(hidden_dims[i]),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1],
                            kernel_size=kernel_sizes[i],
                            stride=strides[i],
                            padding=paddings[i]
                    )
                )
            )
        return nn.Sequential(*modules)


class ResidualBlockA(ResidualBlock):
    kernel_sizes = [3, 3]
    strides = [1, 1]
    paddings = [1, 1]

    def __init__(self, in_channels):

        super(ResidualBlockA, self).__init__()

        self.in_channels = in_channels

        self.conv = self._construct_conv(
                        [self.in_channels] * 3, 
                        self.kernel_sizes, self.strides, self.paddings)
        self.se = SELayer(self.in_channels)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.se(y)
        return x + y
    

class ResidualBlockB(ResidualBlock):
    
    kernel_sizes = [3, 3]
    strides = [2, 1]
    paddings = [1, 1]

    def __init__(self, in_channels):
        super(ResidualBlockB, self).__init__()
        
        self.in_channels = in_channels

        self.conv = self._construct_conv(
                        [self.in_channels, self.in_channels*2, self.in_channels*2],
                        self.kernel_sizes, self.strides, self.paddings)
        self.se = SELayer(self.in_channels * 2)
        self.skip_swish = nn.SiLU()
        self.skip_conv = nn.Conv2d(self.in_channels, self.in_channels * 2,
                                   kernel_size=1,
                                   stride=2,
                                   padding=0
                                )
    
    def forward(self, x):
        y = self.conv(x)
        y = self.se(y)
        x = self.skip_swish(x)
        x = self.skip_conv(x)
        return x + y
