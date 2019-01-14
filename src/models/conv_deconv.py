import sys
import torch
import torch.nn as nn
from torch.nn.functional import relu
import numpy as np

class ConvLayer (nn.Module):
    def __init__(self, c_in, c_out, stride, kernel_size, padding, drop_prob):
        super().__init__()
        self.conv = nn.Conv2d(c_in,c_out,stride=stride, kernel_size=kernel_size, padding=padding).double()
        self.bn = nn.BatchNorm2d(c_out).double()
        self.drop = nn.Dropout2d(p=drop_prob)

    def forward(self, x):
        return self.drop(relu(self.bn(self.conv(x))))

class DeConvLayer(nn.Module):
    """ No RELU BEHIND DECONV DUE TO THE SKIP CONNECTIONS"""
    def __init__(self, c_in, c_out, stride, kernel_size, padding, output_padding, drop_prob):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(c_in, c_out, stride=stride, kernel_size=kernel_size,
                                         padding=padding, output_padding=output_padding).double()
        self.bn = nn.BatchNorm2d(c_out).double()
        self.drop = nn.Dropout2d(p=drop_prob)

    def forward(self, x):
        return self.bn(self.deconv(x))


class ConvDeconv(nn.Module):

    def __init__(self, conv_channels, output_channels=None, strides=None,
                 kernels=None, padding=None, output_padding=None, drop_probs=None,
                 criterion=nn.MSELoss(), optimizer=torch.optim.Adam,
                 learning_rate=0.01, model_name='shallow_model', dropout=0, input_size=(401, 401),
                 add_skip=True, attention_mask='Not', add_skip_at_first=True):
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


        self.num_layers = len(conv_channels)-1
        self.out_channels = output_channels
        self.input_size = input_size

        if strides is None:
            # initialize list with default strides (2,2)
            default_stride = (2,2)
            strides = [default_stride for i in range(self.num_layers)]
        if kernels is None:
            # initialize list with default kernels (7,7)
            default_kernel = (7,7)
            kernels = [default_kernel for i in range(self.num_layers)]
        if padding is None:
            padding = [(2,2) for i in range(self.num_layers)]
        if drop_probs is None:
            drop_probs = [0 for i in range(self.num_layers)]

        dstrides, dkernels, dpadding, output_padding, ddrop_probs = self.compute_strides_and_kernels(strides=strides,
                                                                                        kernels=kernels,
                                                                                        padding=padding,
                                                                                        drop_probs=drop_probs,
                                                                                        output_padding=output_padding,
                                                                                        input_size=self.input_size)


        self.conv_layers = nn.ModuleList([ConvLayer(conv_channels[i], conv_channels[i + 1],
                                                    strides[i], kernels[i], padding=padding[i],
                                                    drop_prob=drop_probs[i])
                                          for i in range(self.num_layers)])

        deconv_channels = conv_channels[::-1]

        if output_channels is not None:
            deconv_channels[-1] = output_channels

        self.deconv_layers = nn.ModuleList([DeConvLayer(deconv_channels[i], deconv_channels[i + 1],
                                                        dstrides[i], dkernels[i],
                                                        padding=dpadding[i],
                                                        drop_prob=ddrop_probs[i],
                                                        output_padding=output_padding[i])
                                            for i in range(self.num_layers)])

        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.train_loss = []
        self.val_loss = []
        self.model_name = model_name

        self.add_skip = add_skip
        self.add_skip_at_first = add_skip_at_first
        self.attention_mask = attention_mask

        if self.add_skip_at_first:
            self.adding_one = 1
        else:
            self.adding_one = 0


    def forward(self, x):

        skip_connection = []
        overlaps = np.exp(np.arange(len(self.conv_layers)) + 1) / np.exp(len(self.conv_layers))
        len(self.conv_layers)


        for i in range(len(self.conv_layers)):
            if (i + self.adding_one)% 2 == 0:
                if i==0 and self.out_channels is not None:
                    skip_connection += [x[:,0:1,:,:]]
                else:
                    skip_connection += [x]

            else:
                skip_connection += [0]

            # use attention mask on forwarded channels
            if self.attention_mask == 'simple':
                if i == 0:
                    zero_one = self.create_zero_one_ratio(shape_tensor=x.shape, ratio_overlap=overlaps[i],
                                                          upper_ratio=0.2, start='simple')
                else:
                    zero_one = self.create_zero_one_ratio(shape_tensor=x.shape, ratio_overlap=overlaps[i],
                                                          upper_ratio=0.2, start='Not')
                    # TODO: Should this tensor be transfered on the gpu?
                x = zero_one * x

            l = self.conv_layers[i]
            x = l(x)


        for i in range(len(self.deconv_layers)):
            l = self.deconv_layers[i]
            skip = skip_connection[len(skip_connection)-1-i]
            x = l(x)
            if self.add_skip:
                x = x + skip
            if i is not len(self.deconv_layers)-1:
                x = relu(x)

        return x

    def compute_strides_and_kernels(self, strides, kernels, padding, drop_probs, input_size=(401,401), output_padding=None, dilation=None):

        if dilation is None:
            dilation = [(1,1) for i in range(len(strides))]

        dstrides = strides[::-1]
        dkernels = kernels[::-1]
        dpadding = padding[::-1]
        ddrop_probs = drop_probs[::-1]

        ddilation = dilation[::-1]
        # now calculate output padding

        if output_padding is None:
            opad = []
        else:
            opad = output_padding
            return dstrides, dkernels, dpadding, opad, ddrop_probs
        # go through all layers and calculate the resulting sizes

        h_in, w_in = input_size
        conv_sizes = [(h_in, w_in)] # tuples containing the resulting size

        for i in range(self.num_layers):
            strd = strides[i]
            kern = kernels[i]
            pad = padding[i]
            dil = dilation[i]

            h_out = int((h_in + 2 * pad[0] - dil[0] * (kern[0] - 1) - 1) / strd[0] + 1)
            w_out = int((w_in + 2 * pad[1] - dil[1] * (kern[1] - 1) - 1) / strd[1] + 1)

            conv_sizes += [(h_out, w_out)]

            h_in = h_out
            w_in = w_out

        # now going through the deconv layers to calculate the potential output sizes of the deconv
        # if they don't match add as much output padding as needed


        for i in range(len(conv_sizes)-1):
            strd = dstrides[i]
            kern = dkernels[i]
            pad = dpadding[i]
            dil = ddilation[i]

            # desired shapes
            h_in, w_in = conv_sizes[len(conv_sizes) - 1 - i]
            h_goal, w_goal = conv_sizes[len(conv_sizes) - 2 - i]
            d_h_out = (h_in - 1) * strd[0] - 2 * pad[0] + kern[0]
            d_w_out = (w_in - 1) * strd[1] - 2 * pad[1] + kern[1]
            opad_h = h_goal - d_h_out
            opad_w = w_goal - d_w_out

            opad += [(opad_h, opad_w)]

        # change returning values
        print('used output padding in this configuration: ', opad)
        return dstrides, dkernels, dpadding, opad, ddrop_probs

    def create_zero_one_ratio(self, shape_tensor, ratio_overlap, upper_ratio, start='Not', ratio_up_to_low_channel=0.5):
        # shape_tensor: (N, C, H, W)
        zero_one = np.zeros(shape_tensor)
        lower_ratio = 1 - upper_ratio + (upper_ratio) * ratio_overlap
        upper_ratio = upper_ratio + (1 - upper_ratio) * ratio_overlap
        num_upper = int(shape_tensor[2] * upper_ratio)
        num_lower = shape_tensor[2] - int(shape_tensor[2] * lower_ratio)
        single_upper = np.zeros((shape_tensor[2], shape_tensor[3]))
        single_lower = np.ones((shape_tensor[2], shape_tensor[3]))
        single_upper[:num_upper, :] = 1.0
        single_lower[:num_lower, :] = 0.0
        if start == 'simple':
            for i in range(shape_tensor[0]):
                zero_one[i, 0, :, :] = single_upper
                zero_one[i, 1, :, :] = single_lower
                zero_one[i, 2, :, :] = np.ones((shape_tensor[2], shape_tensor[3]))
                zero_one[i, 3, :, :] = np.ones((shape_tensor[2], shape_tensor[3]))
        elif start == 'complex':
            for i in range(shape_tensor[0]):
                zero_one[i, 0, :, :] = np.ones((shape_tensor[2], shape_tensor[3]))
                zero_one[i, 1, :, :] = single_upper
                zero_one[i, 2, :, :] = single_lower
                zero_one[i, 3, :, :] = np.ones((shape_tensor[2], shape_tensor[3]))
                zero_one[i, 4, :, :] = np.ones((shape_tensor[2], shape_tensor[3]))
        else:
            num_channel_till_up = int(shape_tensor[1] * ratio_up_to_low_channel)
            for i in range(shape_tensor[0]):
                for j in range(shape_tensor[1]):
                    if j < num_channel_till_up:
                        zero_one[i, j, :, :] = single_upper
                    else:
                        zero_one[i, j, :, :] = single_lower
        zero_one = torch.tensor(zero_one)
        if torch.cuda.is_available():
            zero_one = zero_one.cuda()
        return zero_one
