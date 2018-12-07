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
                 criterion=nn.MSELoss(), optimizer=torch.optim.Adam, learning_rate=0.01, model_name='shallow_model', dropout=0):
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
            default_stride = (2,2)
            strides = [default_stride for i in range(len(conv_channels))]
        if kernels is None:
            # initialize list with default kernels (7,7)
            default_kernel = (7,7)
            kernels = [default_kernel for i in range(len(conv_channels))]
        if padding is None:
            padding = [(3,3) for i in range(len(conv_channels))]

        deconv_strides, deconv_kernels, padding, output_padding = self.compute_strides_and_kernels(strides=strides, kernels=kernels, padding=padding, output_padding=output_padding, input_size=(401,401))

        self.dropout = nn.Dropout2d(p=dropout)

        self.conv_layers = nn.ModuleList([ConvLayer(conv_channels[i], conv_channels[i + 1],
                                                    strides[i], kernels[i])
                                          for i in range(len(conv_channels) - 1)])

        deconv_channels = conv_channels[::-1]
        self.deconv_layers = nn.ModuleList([DeConvLayer(deconv_channels[i], deconv_channels[i + 1],
                                                        deconv_strides[i], deconv_kernels[i],
                                                        output_padding=output_padding[i])
                                            for i in range(len(deconv_channels) - 1)])

        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.train_loss = []
        self.val_loss = []
        self.model_name = model_name

    def forward(self, x):

        skip_connection = []

        x = self.dropout(x)
        print('printing h-shapes now')
        print(x.shape)
        for i in range(len(self.conv_layers)):
            if i%2==0:
                skip_connection += [x]
            else:
                skip_connection += [0]
            l = self.conv_layers[i]
            x = l(x)
            print(x.shape)


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

    def compute_strides_and_kernels(self, strides, kernels, padding, input_size=(401,401), output_padding=None, dilation=None):
        #TODO
        if dilation is None:
            dilation = [(1,1) for i in range(len(strides))]



        dstrides = strides[::-1]
        dkernels = kernels[::-1]
        dpadding = padding[::-1]


        ddilation = dilation[::-1]
        # now calculate output padding

        if output_padding is None:
            opad = []
        else:
            opad = output_padding
            return dstrides, dkernels, dpadding, opad
        # go through all layers and calculate the resulting sizes

        h_in, w_in = input_size
        conv_sizes = [(h_in, w_in)] # tuples containing the resulting size

        for i in range(len(strides)):
            strd = strides[i]
            kern = kernels[i]
            pad = padding[i]
            dil = dilation[i]

            h_out = (h_in + 2 * pad[0] - dil[0] * (kern[0] - 1)) / strd[0] + 1
            w_out = (w_in + 2 * pad[1] - dil[1] * (kern[1] - 1)) / strd[1] + 1

            conv_sizes += [(int(h_out), int(w_out))]

            h_in = h_out
            w_in = w_out

        # now going through the deconv layers to calculate the potential output sizes of the deconv
        # if they don't match add as much output padding as needed

        print((conv_sizes))

        for i in range(len(conv_sizes)-1):
            strd = dstrides[i]
            kern = dkernels[i]
            pad = dpadding[i]
            dil = ddilation[i]

            # desired shapes
            h_in, w_in = conv_sizes[len(conv_sizes) - 1 - i]
            h_goal, w_goal = conv_sizes[len(conv_sizes) - 2 - i]

            print('stride: ', strd[0])
            print('pad: ', pad[0])
            print('kern: ', kern[0])
            d_h_out = (h_in - 1) * strd[0] + 2 * pad[0] - kern[0]
            d_w_out = (w_in - 1) * strd[1] + 2 * pad[1] - kern[1]
            opad_h = h_goal - d_h_out
            opad_w = w_goal - d_w_out

            print('h_in: ' + str(h_in) + '; h_goal: ' + str(h_goal) + '; d_h_out: ' + str(d_h_out))
            print('opad: ', opad_h)
            '''
            if opad_h is not 0:
                print('adding output padding for height of: ', opad_h)

            if opad_w is not 0:
                print('adding output padding for width of: ', opad_w)

            print('appending output padding now')
            print('opad_h: ', opad_h)
            print('opad_w: ', opad_w)
            print(type((opad_h, opad_w)))
            '''
            opad += [(opad_h, opad_w)]

        # change returning values
        print(opad)
        return dstrides, dkernels, dpadding, opad