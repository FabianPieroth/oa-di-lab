import sys
import torch
from torch import nn




class cnn_skipC_model(nn.Module):
    import torch
    from torch import nn

    def __init__(self,
                 criterion=nn.MSELoss(),
                 optimizer=torch.optim.Adam,
                 ic1=1, oc1=32, oc2=64, oc3=128, oc4=256,
                 k_s=(7, 7), stride=2, pad=3,
                 learning_rate=0.01,
                 weight_decay=0,
                 model_name='shallow_model'):

        super(cnn_skipC_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ic1, out_channels=oc1, kernel_size=k_s, stride=stride, padding=pad).double()
        self.conv2 = nn.Conv2d(in_channels=oc1, out_channels=oc2, kernel_size=k_s, stride=stride, padding=pad).double()
        self.conv3 = nn.Conv2d(in_channels=oc2, out_channels=oc3, kernel_size=k_s, stride=stride, padding=pad).double()
        self.conv4 = nn.Conv2d(in_channels=oc3, out_channels=oc4, kernel_size=k_s, stride=stride, padding=pad).double()

        self.deconv1 = nn.ConvTranspose2d(in_channels=oc4, out_channels=oc3, kernel_size=k_s, stride=stride,
                                       padding=pad).double()
        self.deconv2 = nn.ConvTranspose2d(in_channels=oc3, out_channels=oc2, kernel_size=k_s, stride=stride,
                                       padding=pad).double()
        self.deconv3 = nn.ConvTranspose2d(in_channels=oc2, out_channels=oc1, kernel_size=k_s, stride=stride,
                                       padding=pad).double()
        self.deconv4 = nn.ConvTranspose2d(in_channels=oc1, out_channels=ic1, kernel_size=k_s, stride=stride,
                                       padding=pad).double()

        self.relu = torch.nn.functional.relu

        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.train_loss = []
        self.test_loss = []
        self.model_name = model_name
        if torch.cuda.is_available():
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()
            self.conv3 = self.conv3.cuda()
            self.conv4 = self.conv4.cuda()
            self.deconv1 = self.deconv1.cuda()
            self.deconv2 = self.deconv2.cuda()
            self.deconv3 = self.deconv3.cuda()
            self.deconv4 = self.deconv4.cuda()

    def forward(self, X):
        x = self.relu(self.conv1(X))
        x1 = self.relu(self.conv2(x))
        # doing relu before saving the result tfor the skip connection
        x2 = x1.clone()

        x3 = self.relu(self.conv3(x1))
        x3 = self.relu(self.conv4(x3))

        x3 = self.relu(self.deconv1(x3))
        x3 = self.deconv2(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        x5 = self.relu(self.deconv3(x4))
        x5 = self.deconv4(x5)

        out = x5 + X
        out = self.relu(out)
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
                test_loss = self.criterion(test_out, valid_target)
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

