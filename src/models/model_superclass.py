import sys
import torch
import torch.nn as nn
from torch.nn.functional import relu

class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, stride=2, kernel_size=(5,5), padding=2):
        super().__init__()
        self.conv = nn.Conv2d(c_in,c_out,stride=stride, kernel_size=kernel_size, padding=padding).double()
        self.bn = nn.BatchNorm2d(c_out).double()

    def forward(self, x):
        return relu(self.bn(self.conv(x)))

class DeConvLayer(nn.Module):
    """ No RELU BEHIND DECONV DUE TO THE SKIP CONNECTIONS"""
    def __init__(self, c_in, c_out, stride=2, kernel_size=(5,5), padding=2, output_padding=0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(c_in, c_out, stride=stride, kernel_size=kernel_size,
                                         padding=padding, output_padding=output_padding).double()
        self.bn = nn.BatchNorm2d(c_out).double()

    def forward(self, x):
        return self.bn(self.deconv(x))


class ImageTranslator(nn.Module):

    def __init__(self, conv_channels, strides=None, kernels=None, padding=None, output_padding=None,
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

        if strides is None:
            # initialize list with default strides (2,2)
            default_stride = 2
            strides = [default_stride for i in range(len(conv_channels))]
        if kernels is None:
            # initialize list with default kernels (7,7)
            default_kernel = (7,7)
            kernels = [default_kernel for i in range(len(conv_channels))]
        if padding is None:
            padding = [3 for i in range(len(conv_channels))]
        if output_padding is not None:
            self.output_padding = output_padding
        else:
            self.output_padding = [0 for i in range(len(conv_channels))]

        deconv_strides, deconv_kernels, padding, opad = self.compute_strides_and_kernels(strides, kernels, padding)

        self.conv_layers = nn.ModuleList([ConvLayer(conv_channels[i], conv_channels[i + 1],
                                                    strides[i], kernels[i])
                                          for i in range(len(conv_channels) - 1)])

        deconv_channels = conv_channels[::-1]
        self.deconv_layers = nn.ModuleList([DeConvLayer(deconv_channels[i], deconv_channels[i + 1],
                                                        deconv_strides[i], deconv_kernels[i],
                                                        output_padding=self.output_padding[i])
                                            for i in range(len(deconv_channels) - 1)])

        # save parameters
        self.criterion = criterion
        self.kernels = kernels
        self.padding = padding
        self.strides = strides
        
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.train_loss = []
        self.val_loss = []
        self.model_name = model_name

        # write parameters into self to give it to logger

        self.conv_channels = conv_channels
        self.strides = strides
        self.kernels = kernels
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):

        skip_connection = []

        for i in range(len(self.conv_layers)):
            if i%2 == 0:
                skip_connection += [x]
            else:
                skip_connection += [0]
            l = self.conv_layers[i]
            x = l(x)

        for i in range(len(self.deconv_layers)):
            l = self.deconv_layers[i]
            skip = skip_connection[len(skip_connection)-1-i]
            x = l(x) + skip

            if i is not len(self.deconv_layers)-1:
                x = relu(x)

        return x

    def train_model(self, X, y, current_epoch=None):
        """
        Trains the model for one batch
        :param X: batch input
        :param y: batch target
        :param current_epoch: the current epoch for displaying in print statement
        """
        def closure():
            self.optimizer.zero_grad()
            out = self.forward(X)
            loss = self.criterion(out, y)
            if current_epoch is not None:
                sys.stdout.write('\r' + ' epoch ' + str(current_epoch) + ' |  loss : ' + str(loss.item()))
            else:
                sys.stdout.write('\r  loss : ' + str(loss.item()))
            self.train_loss.append(loss.item())
            loss.backward()
            return loss
        self.optimizer.step(closure)

    def set_learning_rate(self, learning_rate):
        """
        setting the learning rate. Can be explicitly done to have another learning rate without new initialization
        :param learning_rate: learning rate to be set
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def compute_strides_and_kernels(self, strides, kernels, padding):
        #TODO
        if self.output_padding is not None:
            opad = [0 for i in range(len(strides))]
        else:
            opad = self.output_padding
        return strides[::-1], kernels[::-1], padding[::-1], opad
