import sys
import torch
import torch.nn as nn
from torch.nn.functional import relu
import numpy as np
import models.utils as ut

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
                 add_skip=True, attention_mask='Not',
                 add_skip_at_first=True, concatenate_skip=False, attention_anchors=None,
                 attention_input_dist=None, attention_network_dist=None, use_upsampling=False, last_kernel_sizes=None,
                 after_skip_channels=None):
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

        self.add_skip_at_first = add_skip_at_first
        self.concatenate_skip = concatenate_skip

        if self.add_skip_at_first:
            self.adding_one = 0
        else:
            self.adding_one = 1

        # increase nr of deconv channels
        deconv_channels_new = deconv_channels.copy()
        if self.concatenate_skip:
            for i in range(len(deconv_channels)-1):
                if (i+self.adding_one) % 2 == 0:
                    deconv_channels_new[i+1] = deconv_channels[i+1] * 2

        self.deconv_layers = nn.ModuleList([DeConvLayer(deconv_channels_new[i], deconv_channels[i + 1],
                                                        dstrides[i], dkernels[i],
                                                        padding=dpadding[i],
                                                        drop_prob=ddrop_probs[i],
                                                        output_padding=output_padding[i])
                                            for i in range(self.num_layers)])

        # using upsampling in the last layer to get higher definition without skip
        self.use_upsampling = use_upsampling



        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.train_loss = []
        self.val_loss = []
        self.model_name = model_name

        self.add_skip = add_skip
        self.attention_mask = attention_mask
        self.attention_anchors = attention_anchors
        self.attention_input_dist = attention_input_dist
        self.attention_network_dist = attention_network_dist
        self.conv_channels = conv_channels
        if use_upsampling:
            print('Using upsampling!')
            self.deconv_layers[-1].stride = 1

        self.last_layers = None
        if after_skip_channels is not None:
            # the first channel of these layers is depending on the amount of masked channels we concat
            # the last channel will be out_channel so the last of deconv_channels
            # the channels in between can be chosen
            # preparing the list to these needs
            after_skip_channels = [np.sum(self.attention_input_dist)+deconv_channels[-1]] \
                                  + after_skip_channels + [deconv_channels[-1]]
            if last_kernel_sizes is None:
                last_kernel_sizes = [(41,11) for _ in range(len(after_skip_channels)+1)]
            self.last_layers = nn.ModuleList()

            for i in range(len(after_skip_channels)-1):
                same_padding_h = int((last_kernel_sizes[i][0]-1)/2)
                same_padding_w = int((last_kernel_sizes[i][1]-1)/2)
                self.last_layers.append(DeConvLayer(after_skip_channels[i], after_skip_channels[i+1],
                                                     stride=1, padding=(same_padding_h,same_padding_w),
                                                     kernel_size=last_kernel_sizes[i],
                                                     output_padding=0, drop_prob=0))

    def forward(self, x):

        skip_connection = []
        overlaps = np.exp(np.arange(len(self.conv_layers)) + 1) / np.exp(len(self.conv_layers))
        len(self.conv_layers)

        for i in range(len(self.conv_layers)):

            # use attention mask on forwarded channels, skips influenced as well
            if self.attention_mask == 'simple':
                if i == 0:
                    zero_one = ut.create_zero_one_ratio(shape_tensor=x.shape, ratio_overlap=overlaps[i],
                                                        upper_ratio=0.2, start='simple',
                                                        attention_anchors=self.attention_anchors,
                                                        attention_input_dist=self.attention_input_dist,
                                                        attention_network_dist=self.attention_network_dist)
                else:
                    zero_one = ut.create_zero_one_ratio(shape_tensor=x.shape, ratio_overlap=overlaps[i],
                                                        upper_ratio=0.2, start='Not',
                                                        attention_anchors=self.attention_anchors,
                                                        attention_input_dist=self.attention_input_dist,
                                                        attention_network_dist=self.attention_network_dist)
                if torch.cuda.is_available():
                    zero_one = zero_one.cuda()
                x = zero_one * x

            if (i + self.adding_one)% 2 == 0:
                if i==0 and self.out_channels is not None and self.add_skip_at_first:
                    skip_connection += [x[:,0:np.sum(self.attention_input_dist),:,:]]
                else:
                    skip_connection += [x]

            else:
                skip_connection += [0]

            l = self.conv_layers[i]
            x = l(x)

        for i in range(len(self.deconv_layers)):
            l = self.deconv_layers[i]
            skip = skip_connection[len(skip_connection)-1-i]
            x = l(x)
            if self.add_skip:
                if self.concatenate_skip:
                    if skip is not 0:
                        x = torch.cat((x, skip), 1)
                else:
                    x = x + skip

            # think about taking this out
            if self.last_layers is not None or i is not len(self.deconv_layers)-1:
                x = relu(x)

        if self.use_upsampling:
            x = torch.nn.functional.interpolate(x, size=(401,401), mode='bilinear')
            if self.concatenate_skip and self.add_skip_at_first:
                if self.last_layers is None:
                    print('you need last layers to fuse the concatenated layers')
                # x = self.last_deconv(x)
        if self.last_layers is not None:
            for i in range(len(self.last_layers)):
                l = self.last_layers[i]
                if i is not len(self.last_layers)-1:
                    x = relu(x)
                x = l(x)
        return x

    def compute_strides_and_kernels(self, strides, kernels, padding, drop_probs, input_size=(401,401),
                                    output_padding=None, dilation=None):

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
