import torch
import torch.nn as nn
from torch.nn.functional import relu

class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, stride, kernel_size, padding, drop_prob):
        super().__init__()
        self.conv = nn.Conv2d(c_in,c_out,stride=stride, kernel_size=kernel_size, padding=padding).double()
        self.bn = nn.BatchNorm2d(c_out).double()
        self.drop = nn.Dropout2d(p=drop_prob)

    def forward(self, x):
        return self.drop(relu(self.bn(self.conv(x))))

class DeConvLayer(nn.Module):
    """NO RELU BEHIND DECONV DUE TO THE SKIP CONNECTIONS"""
    def __init__(self, c_in, c_out, stride, kernel_size, padding, output_padding, drop_prob):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(c_in, c_out, stride=stride, kernel_size=kernel_size,
                                         padding=padding, output_padding=output_padding).double()
        self.bn = nn.BatchNorm2d(c_out).double()
        self.drop = nn.Dropout2d(p=drop_prob)

    def forward(self, x):
        return self.drop(self.bn(self.deconv(x)))


class MaskConvLayer(nn.Module):
    def __init__(self, c_in, c_out, stride, kernel_size, padding, drop_prob):
        super().__init__()
        self.conv = nn.Conv2d(c_in,c_out,stride=stride, kernel_size=kernel_size, padding=padding).double()
        self.bn = nn.BatchNorm2d(c_out).double()
        self.drop = nn.Dropout2d(p=drop_prob)

    def forward(self, x):
        return self.drop(relu(self.bn(self.conv(x))))

class MaskDeConvLayer(nn.Module):
    """NO RELU BEHIND DECONV DUE TO THE SKIP CONNECTIONS"""
    def __init__(self, c_in, c_out, stride, kernel_size, padding, output_padding, drop_prob):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(c_in, c_out, stride=stride, kernel_size=kernel_size,
                                         padding=padding, output_padding=output_padding).double()
        self.bn = nn.BatchNorm2d(c_out).double()
        self.drop = nn.Dropout2d(p=drop_prob)

    def forward(self, x):
        return self.bn(self.deconv(x))


class ConvDeconv(nn.Module):

    def __init__(self, conv_channels, output_channels=None, datatype='homo', strides=None,
                 kernels=None, padding=None, output_padding=None, drop_probs=None,
                 criterion=nn.MSELoss(), optimizer=torch.optim.Adam,
                 learning_rate=0.01, model_name='shallow_model', input_size=(401, 401),
                 input_ss_mask=None, input_ds_mask=None, ds_mask_channels=None):
        """
        initializes net with the specified attributes. The stride, kernels and paddings and output paddings for the
        deconv layers are computed to fit.
        :param conv_channels: output channels of the convolution layers
        :param strides: stride sizes of the conv layers, same dimension as conv_channels
        :param kernels: kernel sizes of the conv layers, same dimension as conv_channels
        """
        super().__init__()


        self.model_file_name = __file__  # save file name to copy file in logger into logging folder
        self.conv_channels = conv_channels
        self.num_layers = len(conv_channels)-1
        self.out_channels = output_channels
        self.input_size = input_size

        self.input_ss_mask = input_ss_mask
        self.input_ds_mask = input_ds_mask

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
        if input_ss_mask is None:
            self.input_ss_mask = [0 for i in range(self.num_layers)]
        if input_ds_mask is None:
            self.input_ds_mask = [0 for i in range(self.num_layers)]

        # the nummber of channels for the ds_mask can be given. Otherwise the sizes are the same
        # as the conv_channels. Also symmetric.
        if ds_mask_channels is None:
            self.ds_mask_channels = self.conv_channels

        dstrides, dkernels, dpadding, output_padding, ddrop_probs = self.compute_strides_and_kernels(strides=strides,
                                                                                        kernels=kernels,
                                                                                        padding=padding,
                                                                                        drop_probs=drop_probs,
                                                                                        output_padding=output_padding,
                                                                                        input_size=self.input_size)

        # multiply the number of input channels with 2 or 3 depending on the fact if we include the masks
        self.conv_layers = nn.ModuleList([ConvLayer(self.input_ds_mask[i] * self.ds_mask_channels[i] +
                                                    self.input_ss_mask[i] + conv_channels[i],
                                                    conv_channels[i + 1],
                                                    strides[i], kernels[i], padding=padding[i],
                                                    drop_prob=drop_probs[i])
                                          for i in range(self.num_layers)])

        deconv_channels = conv_channels[::-1]

        if output_channels is not None:
            deconv_channels[-1] = output_channels

        # get the mask deconv channels
        mask_deconv_channels = self.ds_mask_channels[::-1]

        self.deconv_layers = nn.ModuleList([DeConvLayer(self.input_ds_mask[self.num_layers+i] * mask_deconv_channels[i] +
                                                        self.input_ss_mask[self.num_layers+i] + deconv_channels[i],
                                                        deconv_channels[i + 1],
                                                        dstrides[i], dkernels[i],
                                                        padding=dpadding[i],
                                                        drop_prob=ddrop_probs[i],
                                                        output_padding=output_padding[i])
                                            for i in range(self.num_layers)])


        # initialize mask layers



        self.mask_conv_layers = nn.ModuleList([ConvLayer(self.ds_mask_channels[i], self.ds_mask_channels[i + 1],
                                                         strides[i], kernels[i], padding=padding[i],
                                                         drop_prob=drop_probs[i])
                                               for i in range(self.num_layers)])

        self.mask_deconv_layers = nn.ModuleList([DeConvLayer(mask_deconv_channels[i], mask_deconv_channels[i + 1],
                                                             dstrides[i], dkernels[i],
                                                             padding=dpadding[i],
                                                             drop_prob=ddrop_probs[i],
                                                             output_padding=output_padding[i])
                                                 for i in range(self.num_layers)])

        self.data_type = datatype
        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.train_loss = []
        self.val_loss = []
        self.model_name = model_name

    def forward(self, x):

        # get masks and image from the channels if data type is homo
        # we work with the assumption that the images are in channel 0,
        # ds_mask channel 1, and ss_mask in channel 2
        if self.data_type is 'hetero':
            ds_mask = x[:, 2:3, :, :]   # get the dual speed of sound mask (batch_size, 1, H, W)
            ss_mask = x[:, 1:2, :, :]   # get the singel speed of sound mask (batch_size, 1, H, W)
            x = x[:, 0:1, :, :]         # get the images -> resulting in (batch_size, 1, H, W)
        else:
            ds_mask = 0
            ss_mask = 0

        skip_connection = []

        for i in range(len(self.conv_layers)):

            if i % 2 == 0:
                if i == 0 and self.out_channels is not None:
                    # this should only
                    skip_connection += [x[:, 0:1, :, :]]
                else:
                    skip_connection += [x] #[:, self.conv_channels[i+1], :, :]]

            else:
                skip_connection += [0]

            # prepare the tensor (cat or not)
            if self.data_type is 'hetero':
                x = self.prepare_tensor(im=x, ds=ds_mask, ss=ss_mask, i=i)

            l = self.conv_layers[i]
            x = l(x)

            # now put the masks through the mask layers
            if self.data_type is 'hetero':
                ml = self.mask_conv_layers[i]
                ds_mask = ml(ds_mask)
                #ss_mask = ml(ss_mask)


        for i in range(len(self.deconv_layers)):

            if self.data_type is 'hetero':
                x = self.prepare_tensor(im=x, ds=ds_mask, ss=ss_mask, i=self.num_layers+i)

            l = self.deconv_layers[i]
            skip = skip_connection[len(skip_connection)-1-i]

            x = l(x) + skip
            if i is not len(self.deconv_layers)-1:
                x = relu(x)

            # put masks through deconv layers
            if self.data_type is 'hetero':
                ml = self.mask_deconv_layers[i]
                ds_mask = ml(ds_mask)
                #ss_mask = ml(ss_mask)

        return x

    def prepare_tensor(self, im, ds, ss, i):
        """
        returns the input tensor of the image. Decides based on ds_mask and ss_mask if the masks are concatenated
        :param im: the image (batch_size, C, H, W)
        :param ds: the dual speed of sound mask (batch_size, C_mask, H, W)
        :param ss: the single speed of sound mask (batch_size, 1, H_orig, W_orig)
        :param iterator:
        :return:
        """

        # set the single sample mask to the correct height and width (same value each pixel)
        # shape is batch_size, C, H, W
        _, _, height, width = im.shape
        ss = ss[:, :, 0:height, 0:width]

        print('self.num_layers: ', self.num_layers)
        print('i: ', i)
        if self.input_ss_mask[i] == 1:
            print('        putting ss mask in')
        if self.input_ds_mask[i] == 1:
            print('        putting ds mask in')

        if self.input_ds_mask[i] == 1:
            if self.input_ss_mask[i] == 1:
                # if both masks should be added concatenate the tensor in channel dimension
                x = torch.cat((im, ds, ss), 1)
            else:
                x = torch.cat((im, ds), 1)
        else:
            if self.input_ss_mask[i] == 1:
                x = torch.cat((im, ss), 1)
            else:
                x = im

        return x


    def compute_strides_and_kernels(self, strides, kernels, padding, drop_probs,
                                    input_size=(401,401), output_padding=None, dilation=None):

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