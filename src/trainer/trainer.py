from data.data_loader import ProcessData
from logger.logger import Logger
import numpy as np
from models.model_superclass import ImageTranslator
import torch
import torch.nn as nn
import sys


class CNN_skipCo_trainer(object):

    def __init__(self):

        self.image_type = 'US'

        self.dataset = ProcessData(data_type='homo', train_ratio=0.9, process_raw_data=True,
                                   pro_and_augm_only_image_type=True, do_heavy_augment=False,
                                   do_augment=False, add_augment=True, do_rchannels=True,
                                   do_flip=True, do_blur=True, do_deform=True, do_crop=False,
                                   image_type=self.image_type, get_scale_center=True, single_sample=False,
                                   do_scale_center=False, height_channel_oa=201)

        # TODO: if data_type='hetero' it should not upsample to the same size
        self.model = ImageTranslator(conv_channels=[1, 64, 64, 128, 128, 256, 256, 512],
                                     output_padding=[0, 0, 1, 0, 0, 1, 0],
                                     model_name='deep_2_model')

        self.logger = Logger(model=self.model, project_root_dir=self.dataset.project_root_dir,
                             image_type=self.image_type, dataset=self.dataset)

        if torch.cuda.is_available():
            torch.cuda.current_device()
            self.model.cuda()

        self.batch_size = 1
        self.log_period = 50
        self.epochs = 250
        self.learning_rates = [0 for i in range(self.epochs)]

    def fit(self, learning_rate, lr_method='standard'):
        # get scale and center parameters
        scale_params_low, scale_params_high = self.dataset.load_params(param_type="scale_params")
        mean_image_low, mean_image_high = self.dataset.load_params(param_type="mean_images")


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


        self.learning_rates = self.get_learning_rate(learning_rate, self.epochs, lr_method)

        for e in range(0, self.epochs):
            # setting the learning rate each epoch
            lr = self.learning_rates[e]
            self.model.set_learning_rate(lr)

            # separate names into random batches and shuffle every epoch
            self.dataset.batch_names(batch_size=self.batch_size)

            # in self.batch_number is the number of batches in the training set
            # go through all the batches
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
            if e % self.log_period == 0 or e == self.epochs - 1:
                self.logger.log(save_appendix='_epoch_' + str(e),
                                current_epoch=e,
                                epochs=self.epochs,
                                mean_images=[mean_image_low, mean_image_high],
                                scale_params=[scale_params_low, scale_params_high])

    def predict(self, x):
        return self.model(x)


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

        import math
        self.dataset.batch_names(batch_size=2)
        print('number of files: ', len(self.dataset.train_file_names))
        num = self.dataset.batch_number-1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        self.model.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.; best_loss = 0.; batch_num = 0; losses = []; log_lrs = []
        # in self.batch_number is the number of batches in the training set
        print('batch numbers: ', self.dataset.batch_number)

        for i in range(self.dataset.batch_number):
            sys.stdout.write('\r' + 'current iteration : ' + str(i))

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
            print('lr: ', lr)
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
        #ax.set_xlim((1e-8, 100))
        ax.figure.show()
        ax.figure.savefig('learning_rate_finder.png')
        '''
        return log_lrs, losses


    def get_learning_rate(self, learning_rate, epochs, method):
        """
        Method creating the learning rates corresponding to the corresponding adaptive-method.
        :param learning_rate: base learning rate. Used as max rate for one_cycle and cosine_annealing
        :param epochs: number of epochs the model is trained
        :param method: adaptive-method which should be used: standard, one_cycle, cosine_annealing
        :return: learning_rates as a list
        """
        lrs = []

        if method=='standard' or method is None:
            lrs = [learning_rate for i in range(epochs)]

        elif method=='one_cycle':
            higher_rate = learning_rate
            lower_rate = 1 / 10 * higher_rate

            ann_frac = min(50, int(epochs/3))

            up_num = int((epochs - ann_frac) / 2)
            down_num = up_num
            ann_num = epochs - up_num - down_num

            lr_up = np.linspace(lower_rate, higher_rate, num=up_num)
            lr_down = np.linspace(higher_rate, lower_rate, num=down_num)
            lr_anihilating = np.linspace(lower_rate, 0, num=ann_num)

            lrs = np.append(np.append(lr_up, lr_down), lr_anihilating)

        elif method=='cosine_annealing':
            pass

        return lrs



def main():
    trainer = CNN_skipCo_trainer()
    #trainer.find_lr()
    # fit the first model
    print('\n---------------------------')
    #print(trainer.model)
    print('fitting model')
    trainer.fit(learning_rate=0.0001, lr_method='one_cycle')
    #trainer.predict()
    # torch.save(trainer.model, "../../reports/model.pt")
    # trainer.log_model(model_name=trainer.model.model_name)
    # print('\n---------------------------')

    # print('finding learning rate')
    # trainer.find_lr()


    print('\nfinished')


if __name__ == "__main__":
    main()
