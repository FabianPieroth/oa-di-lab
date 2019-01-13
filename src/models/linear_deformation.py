import torch
import torch.nn as nn
import numpy as np



class DeformationLearner(nn.Module):

    def __init__(self, stride, kernel, padding, input_size=(401,401)):
        super().__init__()

        self.model_filename = __file__
        self.stride = stride
        self.kernel = kernel
        self.padding = padding
        self.input_size = input_size

        # layers
        self.conv = nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=kernel, stride=stride,
                              padding=padding)

        opad = self.calc_output_padding()
        self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1,
                                         kernel_size=kernel, stride=stride,
                                         padding=padding,output_padding=opad)
        conv_h, conv_w = self.calc_conv_output()
        self.linear = nn.Linear(in_features=conv_h*conv_w*3, out_features=conv_h*conv_w)

        # set the parameters of the linear layer to zero except for the diagonal
        self.linear.weight.data = self.linear.weight.data * self.create_beside_diag_identity()
        # dual speed mask layer
        self.mask_conv = nn.Conv2d(in_channels=1, out_channels=1,
                                   kernel_size=kernel, stride=stride,
                                   padding=padding)

    def forward(self, x):
        ds_mask = x[:, 2:3, :, :]   # get the dual   speed of sound mask (batch_size, 1, H, W)
        ss_mask = x[:, 1:2, :, :]   # get the single speed of sound mask (batch_size, 1, H, W)
        image = x[:, 0:1, :, :]         # image

        ds_mask = self.mask_conv(ds_mask)
        ss_mask = ss_mask[:,:,0:134,0:134]
        x = self.conv(image)
        x = torch.cat((x, ds_mask, ss_mask),1)
        x = self.linear(x.view(x.shape[0],-1))
        x = self.deconv(x.view(image.shape))
        return x

    def calc_conv_output(self):
        height, width = self.input_size
        h_conv_out = int((height + 2 * self.padding - (self.kernel - 1) - 1) / self.stride + 1)
        w_conv_out = int((width +  2 * self.padding - (self.kernel - 1) - 1) / self.stride + 1)
        return h_conv_out, w_conv_out

    def calc_output_padding(self):
        height, width = self.input_size
        h_conv_out, w_conv_out = self.calc_output_padding()
        d_h_out = (h_conv_out - 1) * self.stride - 2 * self.padding + self.kernel
        d_w_out = (w_conv_out - 1) * self.stride - 2 * self.padding + self.kernel
        opad_h = height - d_h_out
        opad_w = width  - d_w_out
        return opad_h, opad_w

    def create_beside_diag_identity(self, size, batch_size, beside_diagonal=1, new_axis=0, channel=3):
        ident = np.identity(size)
        for j in range(1, beside_diagonal + 1):
            rng = np.arange(ident.shape[0] - j)
            ident[rng, rng + j] = 1.0
            ident[rng + j, rng] = 1.0
        ident_diag = np.expand_dims(ident, axis=new_axis)
        ident_full = np.zeros((batch_size, ident_diag.shape[0], ident_diag.shape[1], ident_diag.shape[2]))
        for k in range(batch_size):
            ident_full[k, :, :, :] = ident_diag
        ident = ident_full
        for j in range(channel - 1):
            ident = np.append(ident, ident_full, axis=3)

        return torch.tensor(ident)
