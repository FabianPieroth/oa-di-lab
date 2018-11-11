import sys
import torch
from torch import nn




class AwesomeImageTranslator1000(nn.Module):
    import torch
    from torch import nn

    def __init__(self,
                 criterion=nn.MSELoss(),
                 optimizer=torch.optim.Adam,
                 ic1=1, oc1=4, oc2=8, oc3=16, oc4=32, oc5=64,
                 oc6=128, oc7=256, oc8=512, oc9=1024, oc10=2048,
                 k_s=(7, 7), stride=2, pad=3,
                 learning_rate=0.05,
                 weight_decay=0,
                 model_name='deep_model'):

        super(AwesomeImageTranslator1000, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ic1, out_channels=oc1, kernel_size=k_s, stride=stride, padding=pad).double()
        self.conv2 = nn.Conv2d(in_channels=oc1, out_channels=oc2, kernel_size=k_s, stride=stride, padding=pad).double()
        self.conv3 = nn.Conv2d(in_channels=oc2, out_channels=oc3, kernel_size=k_s, stride=stride, padding=pad).double()
        self.conv4 = nn.Conv2d(in_channels=oc3, out_channels=oc4, kernel_size=k_s, stride=stride, padding=pad).double()
        self.conv5 = nn.Conv2d(in_channels=oc4, out_channels=oc5, kernel_size=k_s, stride=stride, padding=pad).double()
        self.conv6 = nn.Conv2d(in_channels=oc5, out_channels=oc6, kernel_size=k_s, stride=stride, padding=pad).double()
        self.conv7 = nn.Conv2d(in_channels=oc6, out_channels=oc7, kernel_size=k_s, stride=stride, padding=pad).double()
        self.conv8 = nn.Conv2d(in_channels=oc7, out_channels=oc8, kernel_size=k_s, stride=stride, padding=pad).double()
        self.conv9 = nn.Conv2d(in_channels=oc8, out_channels=oc9, kernel_size=k_s, stride=stride, padding=pad).double()
        self.conv10 = nn.Conv2d(in_channels=oc9, out_channels=oc10, kernel_size=k_s, stride=stride, padding=pad).double()

        self.deconv1 = nn.ConvTranspose2d(in_channels=oc10, out_channels=oc9, kernel_size=k_s, stride=stride,
                                       padding=pad).double()
        self.deconv2 = nn.ConvTranspose2d(in_channels=oc9, out_channels=oc8, kernel_size=k_s, stride=stride,
                                       padding=3, output_padding=1).double()
        self.deconv3 = nn.ConvTranspose2d(in_channels=oc8, out_channels=oc7, kernel_size=k_s, stride=stride,
                                       padding=pad, output_padding=1).double()
        self.deconv4 = nn.ConvTranspose2d(in_channels=oc7, out_channels=oc6, kernel_size=k_s, stride=stride,
                                       padding=pad).double()
        self.deconv5 = nn.ConvTranspose2d(in_channels=oc6, out_channels=oc5, kernel_size=k_s, stride=stride,
                                          padding=pad).double()
        self.deconv6 = nn.ConvTranspose2d(in_channels=oc5, out_channels=oc4, kernel_size=k_s, stride=stride,
                                          padding=pad, output_padding=1).double()
        self.deconv7 = nn.ConvTranspose2d(in_channels=oc4, out_channels=oc3, kernel_size=k_s, stride=stride,
                                          padding=pad).double()
        self.deconv8 = nn.ConvTranspose2d(in_channels=oc3, out_channels=oc2, kernel_size=k_s, stride=stride,
                                          padding=pad).double()
        self.deconv9 = nn.ConvTranspose2d(in_channels=oc2, out_channels=oc1, kernel_size=k_s, stride=stride,
                                          padding=pad).double()
        self.deconv10 = nn.ConvTranspose2d(in_channels=oc1, out_channels=ic1, kernel_size=k_s, stride=stride,
                                          padding=pad).double()

        self.relu = torch.nn.functional.relu

        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.train_loss = []
        self.test_loss = []
        self.model_name = model_name

    def forward(self, X):
        x1 = self.relu(self.conv1(X))
        x2 = self.relu(self.conv2(x1))
        # doing relu before saving the result before the skip connection
        skip2 = x2.clone()

        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        skip3 = x4.clone()

        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        skip4 = x6.clone()

        x7 = self.relu(self.conv7(x6))
        x8 = self.relu(self.conv8(x7))
        skip5 = x8.clone()

        x9 = self.relu(self.conv9(x8))
        #x10 = self.relu(self.conv10(x9))

        #x11 = self.relu(self.deconv1(x10))
        x12 = self.deconv2(x9)

        x13 = self.relu(x12 + skip5)

        x13 = self.relu(self.deconv3(x13))
        x14 = self.deconv4(x13)

        x15 = self.relu(x14 + skip4)

        x15 = self.relu(self.deconv5(x15))
        x16 = self.deconv6(x15)

        x17 = self.relu(x16 + skip3)

        x17 = self.relu(self.deconv7(x17))
        x18 = self.deconv8(x17)

        x19 = self.relu(x18 + skip2)

        x19 = self.relu(self.deconv9(x19))
        x20 = self.deconv10(x19)

        out = self.relu(x20 + X)

        return out

    def train_model(self, X, y, valid_in=None, valid_target=None, current_epoch=None):

        def closure():
            self.optimizer.zero_grad()
            out = self.forward(X)
            loss = self.criterion(out, y)
            if current_epoch is not None:
                sys.stdout.write('\r' + ' epoch ' + str(current_epoch) + ' |  loss : ' + str(loss.item()))
            else:
                sys.stdout.write('\r  loss : ' + str(loss.item()))
            # print('epoch ' + str(i) + ' |  loss : ' + str(loss.item()))
            self.train_loss.append(loss.item())
            loss.backward()
            return loss

        self.optimizer.step(closure)
        # calculating the test_loss
        if valid_in is not None and valid_target is not None:
            with torch.no_grad():
                test_out = self.forward(valid_in)
                test_loss = self.criterion(test_out.reshape(-1, self.out_size), valid_target)
                self.test_loss.append(test_loss.item())
        return

    def predic(self, X, y):
        return self.forward(X), y


    def train_model_premat(self, X, y, test_in=None, test_target=None, epochs=100):
        """
        legacy method; can be used to train with full data now
        :param X: images input in the format (N, C, H, W)
        :param y: target images in the format (N, C, H, W)
        :param test_in: validation data input
        :param test_target: validation data target
        :param epochs: number of epochs to train
        :return: model is trained
        """
        self.epochs = epochs
        print('start training')
        print('-----------------------------------------')
        for i in range(epochs):
            def closure():
                self.optimizer.zero_grad()
                out = self.forward(X)
                loss = self.criterion(out, y)
                sys.stdout.write('\r' + 'epoch ' + str(i) + ' |  loss : ' + str(loss.item()))
                # print('epoch ' + str(i) + ' |  loss : ' + str(loss.item()))
                self.train_loss.append(loss.item())
                loss.backward()
                return loss

            self.optimizer.step(closure)
            # calculating the test_loss
            if test_in is not None and test_target is not None:
                with torch.no_grad():
                    test_out = self.forward(test_in)
                    test_loss = self.criterion(test_out.reshape(-1, self.out_size), test_target)
                    self.test_loss.append(test_loss.item())
        print('\n-----------------------------------------')
        return

