import numpy as np
import sys
import torch
from torch.nn import Conv2d
from torch.nn import ConvTranspose2d
from torch import nn
from torch.nn.functional import relu

class cnn_skipC_model(nn.Module):

    def __init__(self,
                 criterion=nn.MSELoss(),
                 optimizer=torch.optim.Adam,
                 learning_rate=0.01,
                 weight_decay=0):

        super(AwesomeImageTranslator1000, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=2).double()
        self.deconv = ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=2).double()

        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loss = []
        self.test_loss = []

    def forward(self, X):
        x = self.conv(X)
        x = relu(x)
        x = self.deconv(x)
        # x = relu(x)
        out = x + X
        out = relu(out)
        return out

    def train_model(self, X, y, test_in=None, test_target=None, epochs=100):
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



