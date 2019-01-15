from data.data_loader import ProcessData
from logger.logger_module import Logger
import numpy as np
from models.conv_deconv import ConvDeconv
from models.dilated_conv import DilatedTranslator
from models.model_superclass import ImageTranslator
#from models.SP2_conv_deconv import ConvDeconv
from models.linear_deformation import DeformationLearner
import torch
import random
import torch.nn as nn
import sys


class CNN_skipCo_trainer(object):

    def __init__(self, image_type, batch_size, log_period, epochs, data_type, train_ratio,
                 process_raw_data, pro_and_augm_only_image_type, do_heavy_augment,do_augment,
                 add_augment, do_rchannels,do_flip, do_blur, do_deform, do_crop,do_speckle_noise,
                 trunc_points, get_scale_center, single_sample,do_scale_center,
                 oa_do_scale_center_before_pca,
                 trunc_points_before_pca,
                 oa_do_pca,oa_pca_fit_ratio, oa_pca_num_components,
                 height_channel_oa, use_regressed_oa, include_regression_error, add_f_test,
                 only_f_test_in_target, channel_slice_oa, process_all_raw_folders,
                 conv_channels,kernels, model_name, input_size,output_channels, drop_probs,
                 di_conv_channels, dilations, learning_rates, optimizer, criterion, hetero_mask_to_mask,hyper_no,
                 input_ds_mask, input_ss_mask, ds_mask_channels, attention_mask, add_skip, pca_use_regress,
                 add_skip_at_first):

        self.image_type = image_type

        self.batch_size = batch_size
        self.log_period = log_period
        self.epochs = epochs

        self.dataset = ProcessData(data_type=data_type, train_ratio=train_ratio, process_raw_data=process_raw_data,
                                   pro_and_augm_only_image_type=pro_and_augm_only_image_type,
                                   do_heavy_augment=do_heavy_augment,
                                   do_augment=do_augment, add_augment=add_augment, do_rchannels=do_rchannels,
                                   do_flip=do_flip, do_blur=do_blur, do_deform=do_deform, do_crop=do_crop,
                                   do_speckle_noise=do_speckle_noise,trunc_points=trunc_points,
                                   image_type=image_type, get_scale_center=get_scale_center,
                                   single_sample=single_sample,
                                   do_scale_center=do_scale_center,
                                   trunc_points_before_pca=trunc_points_before_pca,
                                   oa_do_scale_center_before_pca=oa_do_scale_center_before_pca,
                                   oa_do_pca=oa_do_pca, oa_pca_fit_ratio=oa_pca_fit_ratio,
                                   oa_pca_num_components=oa_pca_num_components,
                                   height_channel_oa=height_channel_oa, use_regressed_oa=use_regressed_oa,
                                   include_regression_error=include_regression_error,
                                   add_f_test=add_f_test, only_f_test_in_target=only_f_test_in_target,
                                   channel_slice_oa=channel_slice_oa,
                                   process_all_raw_folders=process_all_raw_folders,
                                   hetero_mask_to_mask=hetero_mask_to_mask,
                                   attention_mask=attention_mask, pca_use_regress=pca_use_regress)

        self.model_convdeconv = ConvDeconv(conv_channels=conv_channels,
                                           #input_ds_mask=input_ds_mask,
                                           #input_ss_mask=input_ss_mask,
                                           #ds_mask_channels=ds_mask_channels,
                                           #datatype=data_type,
                                           kernels=kernels,
                                           model_name=model_name, input_size=input_size,
                                           output_channels=output_channels, drop_probs=drop_probs,
                                           add_skip=add_skip, attention_mask=attention_mask,
                                           add_skip_at_first=add_skip_at_first)

        self.model_dilated = DilatedTranslator(conv_channels=di_conv_channels, dilations=dilations)

        # self.deformation_model = DeformationLearner(stride=3, kernel=3, padding=1)
        self.model = ImageTranslator([self.model_convdeconv])


        # we need optimizer and loss here to not access anything from the model class
        self.model_params = self.model.get_parameters()

        self.optimizer = optimizer
        self.criterion = criterion
        self.model_file_path = self.model.model_file_name
        self.model_name = self.model.model_name

        if torch.cuda.is_available():
            torch.cuda.current_device()
            self.model.cuda()
            # here now wrap the model in Data.Parallel class
            self.model = nn.DataParallel(self.model)

        self.learning_rates = learning_rates

        self.logger = Logger(model=self.model, project_root_dir=self.dataset.project_root_dir,
                             image_type=self.image_type, dataset=self.dataset, batch_size=self.batch_size,
                             epochs=self.epochs,learning_rates=self.learning_rates,hyper_no=hyper_no,
                             model_file_path=self.model_file_path, model_name= self.model_name)




    def fit(self, learning_rate, lr_method='standard'):
        # get scale and center parameters
        scale_params_low, scale_params_high = self.dataset.load_params(param_type="scale_params")
        mean_image_low, mean_image_high = self.dataset.load_params(param_type="mean_images")
        if self.dataset.oa_do_pca:
            # get pca model for logging
            pca_model = self.dataset.load_pca_model()
        else: pca_model = None
        if self.dataset.oa_do_scale_center_before_pca:
            scale_params_low_before_pca, scale_params_high_before_pca = self.dataset.load_params(
                param_type="scale_params_before_pca")
            mean_image_low_before_pca, mean_image_high_before_pca = self.dataset.load_params(
                param_type="mean_images_before_pca")
        else:
            scale_params_low_before_pca, scale_params_high_before_pca = None, None
            mean_image_low_before_pca, mean_image_high_before_pca = None, None

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

        # activate optimizer with the base learning rate
        self.optimizer = self.optimizer(self.model_params, lr=learning_rate)

        # now calculate the learning rates list
        self.learning_rates = self.get_learning_rate(learning_rate, self.epochs, lr_method)

        for e in range(0, self.epochs):
            # setting the learning rate each epoch
            lr = self.learning_rates[e]
            self.optimizer.param_groups[0]['lr'] = lr

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

                # train model one step:
                X = input_tensor
                y = target_tensor



                def closure():
                    self.optimizer.zero_grad()
                    out = self.model(X)
                    loss = self.criterion(out, y)
                    sys.stdout.write('\r' + ' epoch ' + str(e) + ' |  loss : ' + str(loss.item()))
                    self.logger.train_loss.append(loss.item())
                    loss.backward()
                    return loss

                self.optimizer.step(closure)

            # calculate the validation loss and add to validation history
            # self.logger.get_val_loss(val_in=input_tensor_val, val_target=target_tensor_val)

            if input_tensor_val is not None and target_tensor_val is not None:
                with torch.no_grad():
                    val_out = self.model.forward(input_tensor_val)
                    # changed for DGX run
                    val_loss = self.criterion(val_out, target_tensor_val)
                    self.logger.val_loss.append(val_loss.item())

            # save model every x epochs
            if e % self.log_period == 0 or e == self.epochs - 1:
                self.logger.log(save_appendix='_epoch_' + str(e),
                                current_epoch=e,
                                epochs=self.epochs,
                                mean_images=[mean_image_low, mean_image_high],
                                scale_params=[scale_params_low, scale_params_high],
                                mean_images_before_pca=[mean_image_low_before_pca, mean_image_high_before_pca],
                                scale_params_before_pca = [scale_params_low_before_pca, scale_params_high_before_pca],
                                pca_model=pca_model,
                                learning_rates=self.learning_rates)

    def find_lr(self, init_value=1e-8, final_value=10., beta=0.98):
        """
        learning rate finder. goes through multiple learning rates and does one forward
        pass with each and tracks the loss.
        it returns the learning rates and losses so one can make an image of the loss from which one can find
        a suitable learning rate. Pick the highest one on which the loss is still decreasing (so not the minimum)
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
        self.optimizer.param_groups[0]['lr'] = lr
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
            self.optimizer.zero_grad()
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

        if method == 'standard' or method is None:
            lrs = [learning_rate for i in range(epochs)]

        elif method == 'one_cycle':
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

    image_type = 'US'
    batch_size = 32
    log_period = 30
    epochs = 90

    # dataset parameters

    data_type = 'hetero'
    train_ratio = 0.9
    process_raw_data = False
    pro_and_augm_only_image_type = True

    do_heavy_augment = False
    do_augment = False
    add_augment = False
    do_rchannels = False
    do_flip = True
    do_blur = False
    do_deform = False
    do_crop = False
    do_speckle_noise = False
    trunc_points = (0, 1)
    trunc_points_before_pca = (0.0001, 0.9999)
    get_scale_center = True
    single_sample = False
    do_scale_center = True
    oa_do_scale_center_before_pca = False
    oa_do_pca = False
    oa_pca_fit_ratio = 1 # percentage of the train data files to sample for fitting the pca
    oa_pca_num_components = 7
    pca_use_regress = False
    height_channel_oa = 201
    use_regressed_oa = False
    include_regression_error = False
    add_f_test = False
    only_f_test_in_target = False
    channel_slice_oa = None  # [0, 3, 6, 10, 15, 23, 27]
    process_all_raw_folders = True
    hetero_mask_to_mask = False
    add_skip = True
    add_skip_at_first = False

    attention_mask = 'simple'  # 'simple', 'Not', to come: 'complex'

    # model parameters

    # conv_channels = [7, 64, 128, 256, 512, 1024]
    conv_channels = [4, 64, 128, 256, 512, 1024]
    kernels = [(7, 7) for i in range(6)]
    model_name = 'deep_2_model'
    input_size = (401, 401)
    output_channels = 1
    drop_probs = [0 for i in range(6)]

    input_ds_mask = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    input_ss_mask = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    ds_mask_channels = [1, 2, 4, 8, 16, 32]

    learning_rate = 0.0001

    optimizer = torch.optim.Adam
    criterion = nn.MSELoss()

    # dilated model parameters

    di_conv_channels = [1, 64, 64, 64, 64, 64]
    dilations = [1, 2, 4, 8, 16]


    # add hyper parameters for search
    #param_grid = {
    #
    #}

    # number of iterations to be performed for hyperparameter search
    max_evals=1

    # Iterate through the specified number of evaluations
    for i in range(max_evals):

        #params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}

        #print(params)
        trainer = CNN_skipCo_trainer(image_type=image_type, batch_size=batch_size, log_period=log_period,
                                     epochs=epochs, data_type=data_type, train_ratio=train_ratio,
                                     process_raw_data=process_raw_data,
                                     pro_and_augm_only_image_type=pro_and_augm_only_image_type,
                                     do_heavy_augment=do_heavy_augment, do_augment=do_augment,
                                     add_augment=add_augment, do_rchannels=do_rchannels, do_flip=do_flip,
                                     do_blur=do_blur, do_deform=do_deform, do_crop=do_crop,
                                     do_speckle_noise=do_speckle_noise,
                                     trunc_points=trunc_points, get_scale_center=get_scale_center,
                                     trunc_points_before_pca=trunc_points_before_pca,
                                     single_sample=single_sample,
                                     do_scale_center=do_scale_center,
                                     oa_do_scale_center_before_pca=oa_do_scale_center_before_pca,
                                     oa_do_pca=oa_do_pca, oa_pca_fit_ratio=oa_pca_fit_ratio, oa_pca_num_components = oa_pca_num_components,
                                     height_channel_oa=height_channel_oa, conv_channels=conv_channels, kernels=kernels,
                                     model_name=model_name, input_size=input_size, output_channels=output_channels,
                                     input_ss_mask=input_ss_mask, input_ds_mask=input_ds_mask,
                                     ds_mask_channels=ds_mask_channels,
                                     drop_probs=drop_probs,
                                     di_conv_channels=di_conv_channels, dilations=dilations,
                                     optimizer=optimizer, criterion=criterion,
                                     learning_rates=learning_rate,
                                     use_regressed_oa=use_regressed_oa,
                                     include_regression_error=include_regression_error,
                                     add_f_test=add_f_test, only_f_test_in_target=only_f_test_in_target,
                                     channel_slice_oa=channel_slice_oa, process_all_raw_folders=process_all_raw_folders,
                                     hetero_mask_to_mask=hetero_mask_to_mask, hyper_no=i,
                                     attention_mask=attention_mask, add_skip=add_skip,
                                     add_skip_at_first=add_skip_at_first,
                                     pca_use_regress=pca_use_regress
                                     )

        # fit the first model
        print('\n---------------------------')
        print('fitting model')
        trainer.fit(learning_rate=0.0001, lr_method='one_cycle')

    print('\nfinished')


if __name__ == "__main__":
    main()
