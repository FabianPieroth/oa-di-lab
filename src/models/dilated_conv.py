import sys
import torch
import torch.nn as nn
from torch.nn.functional import relu

class DilatingLayer(nn.Module):
    def __init__(self, c_in, c_out, dilation=1, stride=1, kernel_size=(3,3), padding=0):
        super().__init__()
        self.conv = nn.Conv2d(c_in,c_out,stride=stride, dilation=dilation, kernel_size=kernel_size, padding=padding).double()
        self.bn = nn.BatchNorm2d(c_out).double()

    def forward(self, x):
        return relu(self.bn(self.conv(x)))


class DilatedTranslator(nn.Module):

    def __init__(self, conv_channels, dilations, kernels=None,
                 criterion=nn.MSELoss(), optimizer=torch.optim.Adam, learning_rate=0.01, model_name='shallow_model'):
        """
        initializes net with the specified attributes. The stride, kernels and paddings and output paddings for the
        deconv layers are computed to fit.
        :param conv_channels: output channels of the convolution layers
        :param deconv_channels:
        :param strides: stride sizes of the conv layers, same dimension as conv_channels
        :param kernels: kernel sizes of the conv layers, same dimension as conv_channels
        """
        super().__init__()


        self.model_file_name = __file__  # save file name to copy file in logger into logging folder

        if kernels is None:
            # initialize list with default kernels (7,7)
            default_kernel = (3,3)
            kernels = [default_kernel for i in range(len(conv_channels))]


        self.padding = self.compute_padding(dilations, kernels)

        self.conv_layers = nn.ModuleList([DilatingLayer(conv_channels[i], conv_channels[i + 1], dilation=dilations[i],
                                                        kernel_size=kernels[i], padding=self.padding[i])
                                          for i in range(len(conv_channels) - 1)])

        self.last_layer = nn.Conv2d(in_channels=conv_channels[len(conv_channels)-1], out_channels=conv_channels[0],
                                    stride=1, kernel_size=(3,3), padding=1).double()
        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.train_loss = []
        self.val_loss = []
        self.model_name = model_name

    def forward(self, x):
        for l in self.conv_layers:
            x = l(x)
        x = self.last_layer(x)
        return x


    def compute_padding(self, dilation, kernels):
        padding = [int(0.5 * dilation[i] * (kernels[i][0] - 1)) for i in range(len(dilation))]
        return padding