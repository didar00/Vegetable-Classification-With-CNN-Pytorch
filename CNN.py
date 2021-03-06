from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch import nn

class CNN(Module):
    def __init__(self, numChannels, classes, residual=False):
        # call the parent constructor
        super(CNN, self).__init__()
        self.residual = residual
        # initialize the first CONV layer
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=6,
            kernel_size=(3, 3))
        self.relu1 = ReLU()
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=6, out_channels=6,
            kernel_size=(3, 3))
        self.relu2 = ReLU()

        #3
        self.conv3 = Conv2d(in_channels=6, out_channels=16,
            kernel_size=(3, 3))
        self.relu3 = ReLU()

        #4
        self.conv4 = Conv2d(in_channels=16, out_channels=16,
            kernel_size=(3, 3))
        self.relu4 = ReLU()

        #5
        self.conv5 = Conv2d(in_channels=16, out_channels=16,
            kernel_size=(3, 3))
        self.relu5 = ReLU()
        
        self.fc1 = Linear(in_features=102*102*16, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)
        self.blocks = nn.Identity()

    def forward(self, x):
        residual = x
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        if self.residual:
            residual = self.blocks(x)
            x = x.clone() + residual
        
        x = flatten(x, 1)
        x = self.fc1(x)
        
        output = self.logSoftmax(x)
        # return the output predictions
        return output