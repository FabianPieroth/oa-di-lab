from data.data_loader import ProcessData
from logger.logger_module import Logger
import numpy as np
from models import deep_model
import torch
import torch.nn as nn
import sys


class CNN_skipCo_trainer(object):
    def __init__(self):

        self.image_type = 'US'

        self.dataset = ProcessData(train_ratio=0.9, process_raw_data=False,
                                   do_augment=False, add_augment=False,
                                   do_flip=True, do_blur=True, do_deform=True, do_crop=True,
                                   image_type=self.image_type, get_scale_center=False, single_sample=True)

        self.model = deep_model.deep_model(
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam,
            learning_rate=0.001,
            weight_decay=0
        )

        if torch.cuda.is_available():
            torch.cuda.current_device()
            self.model.cuda()

        self.logger = Logger(model=self.model, project_root_dir=self.dataset.project_root_dir,
                             image_type=self.image_type)
        self.epochs = 250

    def fit(self, learning_rate, use_one_cycle=False):
        # get scale and center parameters
        scale_params_low, scale_params_high = self.dataset.load_params(param_type="scale_params")
        mean_image_low, mean_image_high = self.dataset.load_params(param_type="mean_images")

        # currently for one image:
        '''
        self.dataset.batch_names(batch_size=5)
        X, Y = self.dataset.create_train_batches(self.dataset.train_batch_chunks[1])
        print(X.shape)
        X = X[0,:,:]
        Y = Y[0,:,:]
        '''
        # load validation set, normalize and parse into tensor

        input_tensor_val, target_tensor_val = self.dataset.scale_and_parse_to_tensor(
                                                    batch_files=self.dataset.val_file_names,
                                                    scale_params_low=scale_params_low,
                                                    scale_params_high=scale_params_high,
                                                    mean_image_low=mean_image_low,
                                                    mean_image_high=mean_image_high)

        if torch.cuda.is_available():
            input_tensor_val = input_tensor_val.cuda()
            target_tensor_val = target_tensor_val.cuda()

        if use_one_cycle:
            higher_rate = learning_rate
            lower_rate = 1 / 10 * higher_rate

            num_epochs = self.epochs
            up_num = int((num_epochs-50)/2)
            down_num = up_num
            ann_num = num_epochs - up_num - down_num
            lr_up = np.linspace(lower_rate, higher_rate, num=up_num)
            lr_down = np.linspace(higher_rate, lower_rate, num=down_num)
            lr_anihilating = np.linspace(lower_rate, 0, num=ann_num)
            learning_rates = np.append(np.append(lr_up, lr_down), lr_anihilating)

        else:
            self.model.set_learning_rate(learning_rate)

        
        for e in range(0, self.epochs):
            if use_one_cycle:
                lr = learning_rates[e]
                self.model.set_learning_rate(lr)
            # separate names into random batches and shuffle every epoch
            self.dataset.batch_names(batch_size=32)

            # in self.batch_number is the number of batches in the training set
            for i in range(self.dataset.batch_number):
                input_tensor, target_tensor = self.dataset.scale_and_parse_to_tensor(
                                                batch_files=self.dataset.train_batch_chunks[i],
                                                scale_params_low=scale_params_low,
                                                scale_params_high=scale_params_high,
                                                mean_image_low=mean_image_low,
                                                mean_image_high=mean_image_high)

                if torch.cuda.is_available():

                    input_tensor = input_tensor.cuda()
                    target_tensor = target_tensor.cuda()

                self.model.train_model(input_tensor, target_tensor, current_epoch=e)

            # calculate the validation loss and add to validation history
            self.logger.get_val_loss(val_in=input_tensor_val, val_target=target_tensor_val)
            # save model every x epochs
            if e % 25 == 0 or e == self.epochs - 1:
                self.logger.log(save_appendix='_epoch_' + str(e),
                                current_epoch=e,
                                epochs=self.epochs,
                                mean_images=[mean_image_low, mean_image_high],
                                scale_params=[scale_params_low, scale_params_high])

                # how to undo the scaling:
                # unscaled_X = utils.scale_and_center_reverse(scale_center_X,
                #  scale_params_low, mean_image_low, image_type = self.dataset.image_type)
                # unscaled_Y = utils.scale_and_center_reverse(scale_center_Y, scale_params_high,
                #  mean_image_high, image_type=self.dataset.image_type)
    def predict(self):
        # self.model.predict()

        # see self.dataset.X_val and self.dataset.Y_val
        pass


    def find_lr(self, init_value=1e-8, final_value=10., beta=0.98):
        """
        learning rate finder. goes through multiple learning rates and does one forward pass with each and tracks the loss.
        it returns the learning rates and losses so one can make an image of the loss from which one can find a suitable learning rate. Pick the highest one on which the loss is still decreasing (so not the minimum)
        :param train_in: training data input
        :param target: training data target
        :param init_value: inital value which the learning rate start with. init_value < final_value. Default: 1e-8
        :param final_value: last value of the learning rate. Default: 10.
        :param beta: used for smoothing the loss. Default: 0.98
        :return: log_learning_rates, losses
        """

        # get scale and center parameters
        scale_params_low, scale_params_high = self.dataset.load_params(param_type="scale_params")
        mean_image_low, mean_image_high = self.dataset.load_params(param_type="mean_images")

        from torch.autograd import variable as Variable
        import math
        print(len(self.dataset.train_file_names))
        num = len(self.dataset.train_file_names)-1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        self.model.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        #for i in range(1, train_in.shape[0]):
        self.dataset.batch_names(batch_size=2)
        # in self.batch_number is the number of batches in the training set
        print(self.dataset.batch_number)
        for i in range(self.dataset.batch_number):
            sys.stdout.write('\r  current iteration : ' + str(i))

            input_tensor, target_tensor = self.dataset.scale_and_parse_to_tensor(
                batch_files=self.dataset.val_file_names,
                scale_params_low=scale_params_low,
                scale_params_high=scale_params_high,
                mean_image_low=mean_image_low,
                mean_image_high=mean_image_high)

            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()

            batch_num += 1
            # As before, get the loss for this mini-batch of inputs/outputs
            self.model.optimizer.zero_grad()
            outputs = self.model.forward(input_tensor)
            loss = self.model.criterion(outputs, target_tensor)
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                print('loss exploding')
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            # Do the SGD step
            loss.backward()
            self.model.optimizer.step()
            # Update the lr for the next step
            lr *= mult
            self.model.optimizer.param_groups[0]['lr'] = lr

        # Plot the results

        '''
        lrs = 10 ** np.array(log_lrs)
        fig, ax = plt.subplots(1)
        ax.plot(lrs, losses)
        ax.set_xscale('log')
        ax.set_xlim((1e-8, 1))
        ax.figure.show()
        ax.figure.savefig('learning_rate_finder.png')
        '''
        return log_lrs, losses

def main():
    trainer = CNN_skipCo_trainer()

    # fit the first model
    print('---------------------------')
    print('fitting deep model')
    trainer.fit(learning_rate=0.01, use_one_cycle=True)
    trainer.predict()
    # torch.save(trainer.model, "../../reports/model.pt")
    # trainer.log_model(model_name=trainer.model.model_name)
    #print('\n---------------------------')

    #print('finding learning rate')
    #trainer.find_lr()


    print('\nfinished')

if __name__ == "__main__":
    main()
