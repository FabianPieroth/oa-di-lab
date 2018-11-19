import sys
import torch
from torch import nn
import numpy as np

class shallow_model_u_bn(nn.Module):
    import torch
    from torch import nn

    def __init__(self,
                 criterion=nn.MSELoss(),
                 optimizer=torch.optim.Adam,
                 ic1=1, oc1=32, oc2=64, oc3=64, oc4=128,
                 k_s=(7, 7), stride1=1, stride2=2, pad=3,
                 learning_rate=0.01,
                 weight_decay=0,
                 model_name='shallow_model_u_bn'):

        super(shallow_model_u_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ic1, out_channels=oc1, kernel_size=k_s, stride=stride1, padding=pad).double()
        self.conv2 = nn.Conv2d(in_channels=oc1, out_channels=oc2, kernel_size=k_s, stride=stride2, padding=pad).double()
        self.conv3 = nn.Conv2d(in_channels=oc2, out_channels=oc3, kernel_size=k_s, stride=stride1, padding=pad).double()
        self.conv4 = nn.Conv2d(in_channels=oc3, out_channels=oc4, kernel_size=k_s, stride=stride2, padding=pad).double()

        self.deconv1 = nn.ConvTranspose2d(in_channels=oc4, out_channels=oc3, kernel_size=k_s, stride=stride1,
                                          padding=pad).double()
        self.deconv2 = nn.ConvTranspose2d(in_channels=oc3, out_channels=oc2, kernel_size=k_s, stride=stride2,
                                          padding=pad).double()
        self.deconv3 = nn.ConvTranspose2d(in_channels=oc2, out_channels=oc1, kernel_size=k_s, stride=stride1,
                                          padding=pad).double()
        self.deconv4 = nn.ConvTranspose2d(in_channels=oc1, out_channels=ic1, kernel_size=k_s, stride=stride2,
                                          padding=pad).double()

        self.bn0 = nn.BatchNorm2d(num_features=ic1).double()
        self.bn1 = nn.BatchNorm2d(num_features=oc1).double()
        self.bn2 = nn.BatchNorm2d(num_features=oc2).double()
        self.bn3 = nn.BatchNorm2d(num_features=oc3).double()
        self.bn4 = nn.BatchNorm2d(num_features=oc4).double()

        self.bn5 = nn.BatchNorm2d(num_features=oc3).double()
        self.bn6 = nn.BatchNorm2d(num_features=oc2).double()
        self.bn7 = nn.BatchNorm2d(num_features=oc1).double()
        self.bn8 = nn.BatchNorm2d(num_features=ic1).double()

        self.relu = torch.nn.functional.relu

        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.train_loss = []
        self.val_loss = []
        self.model_name = model_name

        self.model_file_name = __file__  # save file name to copy file in logger into logging folder

    def forward(self, X):
        x = self.relu(self.bn1(self.conv1(X)))
        x1 = self.relu(self.bn2(self.conv2(x)))
        # doing relu before saving the result tfor the skip connection
        x2 = x1.clone()

        x3 = self.relu(self.bn3(self.conv3(x1)))
        x3 = self.relu(self.bn4(self.conv4(x3)))

        x3 = self.relu(self.bn5(self.deconv1(x3)))
        x3 = self.bn6(self.deconv2(x3))

        x4 = x2 + x3
        x4 = self.relu(x4)

        x5 = self.relu(self.bn7(self.deconv3(x4)))
        x5 = self.bn8(self.deconv4(x5))

        out = x5 + X
        out = self.relu(out)
        return out

    def train_model(self, X, y, current_epoch=None):

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
        return

    def predic(self, X, y):
        return self.forward(X), y

    def get_val_loss(self, val_in=None, val_target=None):
        # do a forward pass with validation set for every epoch and get validation loss
        if val_in is not None and val_target is not None:
            with torch.no_grad():
                val_out = self.forward(val_in)
                val_loss = self.criterion(val_out, val_target)
                self.val_loss.append(val_loss.item())
        return


    def set_learning_rate(self, learning_rate):
        """
        setting the learning rate. Can be explicitly done to have another learning rate without new initialization
        :param learning_rate: learning rate to be set
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate


