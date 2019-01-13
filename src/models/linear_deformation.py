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
                              padding=padding).double()

        opad = self.calc_output_padding()
        self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1,
                                         kernel_size=kernel, stride=stride,
                                         padding=padding,output_padding=opad).double()
        conv_h, conv_w = self.calc_conv_output()
        self.linear = nn.Linear(in_features=conv_h*conv_w*3, out_features=conv_h*conv_w).double()

        # set the parameters of the linear layer to zero except for the diagonal
        # assumption conv_h = conv_w
        print(self.linear.weight.data.shape)
        self.linear.weight.data = self.linear.weight.data * self.create_beside_diag_identity(conv_h**2).double()
        print('done')
        # dual speed mask layer
        self.mask_conv = nn.Conv2d(in_channels=1, out_channels=1,
                                   kernel_size=kernel, stride=stride,
                                   padding=padding).double()

    def forward(self, x):
        ds_mask = x[:, 2:3, :, :]   # get the dual   speed of sound mask (batch_size, 1, H, W)
        ss_mask = x[:, 1:2, :, :]   # get the single speed of sound mask (batch_size, 1, H, W)
        image = x[:, 0:1, :, :]         # image
        import matplotlib.pyplot as plt
        plt.imshow(image[0,0,:,:])
        plt.show()
        ds_mask = self.mask_conv(ds_mask)
        ss_mask = ss_mask[:,:,0:134,0:134]
        downscaled_image = self.conv(image)
        x = torch.cat((downscaled_image, ds_mask, ss_mask),1)

        x = self.linear(x.view(x.shape[0],-1))
        plt.imshow(x.view(downscaled_image.shape)[0,0,:,:])
        plt.show()
        x = self.deconv(x.view(downscaled_image.shape))
        return x

    def calc_conv_output(self):
        height, width = self.input_size
        h_conv_out = int((height + 2 * self.padding - (self.kernel - 1) - 1) / self.stride + 1)
        w_conv_out = int((width +  2 * self.padding - (self.kernel - 1) - 1) / self.stride + 1)
        return h_conv_out, w_conv_out

    def calc_output_padding(self):
        height, width = self.input_size
        h_conv_out, w_conv_out = self.calc_conv_output()
        d_h_out = (h_conv_out - 1) * self.stride - 2 * self.padding + self.kernel
        d_w_out = (w_conv_out - 1) * self.stride - 2 * self.padding + self.kernel
        opad_h = height - d_h_out
        opad_w = width  - d_w_out
        return opad_h, opad_w

    def create_beside_diag_identity(self, size, beside_diagonal=1, new_axis=0, channel=3):
        ident = np.identity(size)
        for j in range(1, beside_diagonal + 1):
            rng = np.arange(ident.shape[0] - j)
            ident[rng, rng + j] = 1.0
            ident[rng + j, rng] = 1.0
        ident_diag = ident
        for j in range(channel - 1):
            ident = np.append(ident, ident_diag, axis=1)

        return torch.tensor(ident)
