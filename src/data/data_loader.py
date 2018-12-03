import os
import pickle
import sys
from pathlib import Path
import numpy as np
import random
import data.data_processing as dp
import torch


class ProcessData(object):
    """
    TODO: Description of data loader in general; keep it growing
    train_ratio:    Gives the train and validation split for the model
    data_type:      String which Data set should be processed (at the moment: 'homo' or 'hetero')
    process_raw:    Process the raw data in the input folder and load them into processed folder
    image_type:     Either 'US' or 'OA' to select which data should be loaded
    do_augment:     Do the augmentation and save them in the corresponding folders
    """

    def __init__(self,
                 train_ratio,
                 image_type,
                 data_type,
                 single_sample=False,
                 add_augment=True,
                 do_augment=False,
                 do_heavy_augment=False,
                 process_raw_data=False,
                 pro_and_augm_only_image_type=False,
                 height_channel_oa=201,
                 do_flip=True,
                 do_deform=True,
                 num_deform=3,
                 do_blur=True,
                 do_crop=True,
                 do_rchannels=True,
                 num_rchannels=2,
                 get_scale_center=True,
                 do_scale_center=True,
                 trunc_points = (0.0001, 0.9999),
                 logger_call=False):

        # initialize and write into self, then call the prepare data and return the data to the trainer
        self.train_ratio = train_ratio
        self.data_type = data_type
        self.accepted_data_types = ['homo', 'hetero']

        if self.data_type not in self.accepted_data_types:
            sys.exit('No acceptable data_type selected!')

        self.do_augment = do_augment  # call data_augment method
        self.do_flip = do_flip
        self.do_deform = do_deform
        self.num_deform = num_deform
        self.do_blur = do_blur
        self.do_crop = do_crop
        self.do_rchannels = do_rchannels
        self.num_rchannels = num_rchannels

        self.height_channel_oa = height_channel_oa

        # the order of the following two lists has to be the same and has to be extended if new augmentations are done
        self.all_augment = ['flip', 'deform', 'blur', 'crop', 'rchannels']
        self.bool_augment = [self.do_flip, self.do_deform, self.do_blur, self.do_crop, self.do_rchannels]
        self.names_of_augment = [self.all_augment[i] for i in range(len(self.all_augment)) if self.bool_augment[i]]

        self.add_augment = add_augment  # bool if augmented data should be included in training
        self.do_heavy_augment = do_heavy_augment

        self.image_type = image_type
        self.single_sample = single_sample  # if this is True only a single image will be loaded in the batch (dev)

        self.project_root_dir = str(Path().resolve().parents[1])  # root directory
        # self.project_root_dir = '/mnt/local/mounted'

        self.dir_raw_in = self.project_root_dir + '/data' + '/' + self.data_type + '/raw' + '/new_in'
        self.dir_processed = self.project_root_dir + '/data' + '/' + self.data_type + '/processed'
        self.dir_processed_all = self.dir_processed + '/processed_all'
        self.dir_augmented = self.project_root_dir + '/data' + '/' + self.data_type + '/processed' + '/augmented'
        self.all_folder = False  # check if raw folder was already processed
        self.pro_and_augm_only_image_type = pro_and_augm_only_image_type
        self.process_oa = not self.pro_and_augm_only_image_type or \
                          (self.pro_and_augm_only_image_type and self.image_type == 'OA')  # process raw oa data
        self.process_us = not self.pro_and_augm_only_image_type or \
                          (self.pro_and_augm_only_image_type and self.image_type == 'US')  # process raw us data
        self.process_raw = process_raw_data  # call method _process_raw_data
        self.get_scale_center = get_scale_center  # get scaling and mean image and store them
        self.do_scale_center = do_scale_center  # applies scale and center to the data
        self.trunc_points = trunc_points # quantiles at which to truncate OA data
        self.dir_params = self.project_root_dir + '/data' + '/' + self.data_type + '/params'
        self.set_random_seed = 42  # set a random seed to enable reproducable samples

        self.logger_call = logger_call  # initialise the data_loader but do nothing else

        # run _prepare_data which calls the methods for preparation, also augmentation etc.
        if not self.logger_call:
            self._prepare_data()

        # self.X_train, self.y_train, self.X_test, self.y_test, self.df_complete = self._prepare_data()

    def _prepare_data(self):
        # process data
        if self.process_raw:
            self._process_raw_data()

        if self.do_augment:
            self._augment_data()

        # get the original file names, split them up to validation and training and write them into self
        self.original_file_names = self._retrieve_original_file_names()
        if self.single_sample:
            self.train_file_names = [self.original_file_names[0]]
            self.val_file_names = [self.original_file_names[0]]
        else:
            self.train_file_names, self.val_file_names = self._train_val_split(
                original_file_names=self.original_file_names)
            self._add_augmented_file_names_to_train()
            self.train_file_names = self._delete_val_from_augmented(val_names=self.val_file_names,
                                                                    train_names=self.train_file_names)
        if self.get_scale_center:
            self._get_scale_center()

        print('There are ' + str(len(self.train_file_names)) + ' files in the training set')

        print('There are ' + str(len(self.val_file_names)) + ' files in the validation set')

        self.X_val, self.Y_val = self._load_processed_data(full_file_names=self.val_file_names)

    ##################################################################
    # ###### Data Loading and Preparation ############################
    ##################################################################

    def batch_names(self, batch_size):
        # shuffle the train_file_names; this gets called every epoch
        self.set_random_seed = self.set_random_seed + 1
        random.seed(self.set_random_seed)
        if len(self.train_file_names) == 0:
            sys.exit('There are no files in the training set,' +
                     ' consider changing training_ratio or you have to debug.')
        self.train_file_names = random.sample(self.train_file_names, len(self.train_file_names))

        if self.single_sample:
            batch_size = 1  # if only signle sample is called, set batch_size on 1

        # give a list and return the corresponding batch names
        self.train_batch_chunks = np.array_split(np.array(self.train_file_names),
                                                 int(len(self.train_file_names) / batch_size))
        self.batch_number = len(self.train_batch_chunks)

    def create_train_batches(self, batch_names):
        # return the batches
        x, y = self._load_processed_data(full_file_names=batch_names)
        return x, y

    def _process_raw_data(self):
        # load the raw data in .mat format, split up the us and oa and load them in dictionaries into processed folder
        in_directories = dp.ret_all_files_in_folder(folder_path=self.dir_raw_in, full_names=False)
        print('Process raw data')
        if not self.all_folder:
            skip_dirs = []
            for sub in in_directories:
                if (next((True for s in os.listdir(self.dir_processed_all + '/ultrasound') if sub in s), False)
                        and next((True for s in os.listdir(self.dir_processed_all + '/optoacoustic') if sub in s),
                                 False)):
                    skip_dirs.append(sub)
                    print('As preprocessed data already exist, skip Folder:' + sub)
            # skip already processed folders
            in_directories = list(set(skip_dirs) ^ set(in_directories))

        for chunk_folder in in_directories:
            sample_directories = dp.ret_all_files_in_folder(folder_path=self.dir_raw_in + '/' + chunk_folder,
                                                            full_names=False)
            print('Processing data from raw input folder: ' + chunk_folder)
            np.random.seed(self.set_random_seed)  # at the moment set_random_seed = 42
            for sample_folder in sample_directories:
                if self.data_type == self.accepted_data_types[0]:
                    in_files = dp.ret_all_files_in_folder(folder_path=self.dir_raw_in + '/' +
                                                                      chunk_folder + '/' + sample_folder,
                                                            full_names=False)
                    us_file = [s for s in in_files if 'US_' in s]
                    oa_file = [s for s in in_files if 'OA_' in s]

                    if us_file and self.process_us:
                        dp.pre_us_homo(new_in_folder=self.dir_raw_in, study_folder=chunk_folder,
                                       filename=us_file[0], scan_num=sample_folder, save_folder=self.dir_processed_all)

                    if oa_file and self.process_oa:
                        dp.pre_oa_homo(new_in_folder=self.dir_raw_in, study_folder=chunk_folder,
                                       filename=oa_file[0], scan_num=sample_folder, save_folder=self.dir_processed_all,
                                       cut_half=True, height_channel=self.height_channel_oa)

                elif self.data_type == self.accepted_data_types[1]:
                    if not self.image_type == 'US':
                        sys.exit('There is only Ultrasound images in the hetero data set.')
                    in_files = dp.ret_all_files_in_folder(folder_path=self.dir_raw_in + '/' +
                                                                      chunk_folder + '/' + sample_folder,
                                                          full_names=False)
                    us_low_samples = [s for s in in_files if 'US_low' in s]
                    us_high_samples = [s for s in in_files if 'US_high' in s]
                    dp.pre_us_hetero(new_in_folder=self.dir_raw_in, study_folder=chunk_folder, scan_num=sample_folder,
                                     filename_low=us_low_samples[0], filename_high=us_high_samples[0],
                                     save_folder=self.dir_processed_all)
                else:
                    print('This should be an empty else, to be stopped before coming here.')

    def _train_val_split(self, original_file_names):
        # this should only be called once at the beginning to ensure the same random seed
        random.seed(self.set_random_seed)
        original_file_names = random.sample(original_file_names, len(original_file_names))
        train_size = int(len(original_file_names) * self.train_ratio)
        self.train_names, val_names = original_file_names[:train_size], original_file_names[train_size:]
        return self.train_names, val_names

    def _retrieve_original_file_names(self):
        # get all the complete file names (path/filename) of the selected data to train on
        if self.image_type not in ['US', 'OA']:
            sys.exit('Error: No valid image_type selected!')
        else:
            if self.image_type == 'US':
                end_folder = 'ultrasound'
            else:
                end_folder = 'optoacoustic'

        if len(dp.ret_all_files_in_folder(folder_path=self.dir_processed_all + '/' + end_folder,
                                          full_names=False)) == 0:
            sys.exit('There are no Files in the processed Folder, please run again with process_raw_data=True.')

        file_names = []
        # original images
        file_names = self._names_to_list(folder_name=self.dir_processed_all + '/' + end_folder, name_list=file_names)
        return file_names

    def _add_augmented_file_names_to_train(self):
        # add the file names of the augmented data to self.train_file_names
        path_augmented = self.dir_processed + '/augmented'
        if self.image_type == 'US':
            end_folder = 'ultrasound'
        else:
            end_folder = 'optoacoustic'

        if self.add_augment:
            for aug_name in self.names_of_augment:
                self.train_file_names = self._names_to_list(
                    folder_name=path_augmented + '/' + aug_name + '/' + end_folder,
                    name_list=self.train_file_names)

    def _delete_val_from_augmented(self, val_names, train_names):
        # deletes the augmented data from the validation set from the training files
        names = [s for s in train_names if not self._detect_val_in_augment(s, val_names)]
        return names

    def _detect_val_in_augment(self, string, val_list):
        contained_in_val = any(self._extract_name_from_path(name) in string for name in val_list)
        return contained_in_val

    def _extract_name_from_path(self, string, without_ch=True):
        # a small helper function to get the file name from the whole path
        # needed because we can't use os.path on server
        filename = ''
        channel = ''
        found_slash = True
        found_underscore = True
        for i in reversed(range(len(string))):
            sub = string[i]
            if sub == '/':
                found_slash = False
            if sub == '_':
                found_underscore = False
            if found_slash:
                filename = sub + filename
            if found_underscore:
                channel = sub + channel
        if without_ch:
            filename = filename[:-len(channel)]
        return filename

    def _names_to_list(self, folder_name, name_list):
        # extract file names from folder and add path name to it
        file_names = dp.ret_all_files_in_folder(folder_path=folder_name, full_names=False)
        # add path to file names and add them to list
        name_list.extend([str(folder_name) + '/' + s for s in file_names])
        return name_list

    def _load_processed_data(self, full_file_names):
        # load the already preprocessed data and store it into X (low) and Y (high) numpy array
        # full_file_names:  iterable list of complete names: so path/filename
        x = np.array(
            [np.array(self._load_file_to_numpy(full_file_name=fname,
                                               image_sign=self.image_type + '_low')) for fname in full_file_names])
        y = np.array(
            [np.array(self._load_file_to_numpy(full_file_name=fname,
                                               image_sign=self.image_type + '_high')) for fname in full_file_names])

        return x, y

    def _load_file_to_numpy(self, full_file_name, image_sign):
        # helper function to load and read the data; pretty inefficient right now
        #  as we need to open every dict two times
        with open(full_file_name, 'rb') as handle:
            sample = pickle.load(handle)
        sample_array = [value for key, value in sample.items() if image_sign in key][0]
        return sample_array

    def _save_params_with_pickle(self, file, folder_name, file_name):
        # use this to save pairs of low and high quality pictures
        with open(self.dir_params + '/' + folder_name + '/' + file_name, 'wb') as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##################################################################
    # ###### Data Augmentation #######################################
    ##################################################################

    def _augment_data(self):
        # set random seed
        random.seed(self.set_random_seed)
        np.random.seed(self.set_random_seed)

        self._clear_aug_directories()
        if self.data_type == self.accepted_data_types[0]:
            if self.image_type == 'OA' and self.do_blur:
                print('No blur augmentation for OA data')

            for end_folder in ['ultrasound', 'optoacoustic']:
                to_be_aug_files = dp.ret_all_files_in_folder(folder_path=self.dir_processed_all + '/' + end_folder,
                                          full_names=True)
                if end_folder == 'ultrasound':
                    file_prefix = 'US'
                else:
                    file_prefix = 'OA'
                if self.pro_and_augm_only_image_type and not self.image_type == file_prefix:
                    continue
                if len(to_be_aug_files) == 0:
                    sys.exit('There are no processed files to be augmented, restart with pre process = True')

                # flip first as some other augmentations are also done on flipped images
                # rchannels also done on same images
                for filename in to_be_aug_files:
                    print('augmenting file', self._extract_name_from_path(filename, without_ch=False))
                    x = self._load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_low')
                    y = self._load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_high')
                    if self.do_flip:
                        dp.do_flip(x=x, y=y, file_prefix=file_prefix,
                                   filename=self._extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder,
                                   path_to_augment=self.dir_augmented)

                    # the additional channels are only taken from the processed_all folder
                    if self.do_rchannels and end_folder == 'ultrasound':
                        dp.do_rchannels(end_folder=end_folder,
                                        filename=self._extract_name_from_path(filename, without_ch=False),
                                        read_in_folder=self.dir_raw_in,
                                        num_channels=self.num_rchannels, path_to_augment=self.dir_augmented)

                    if self.do_blur and end_folder == 'ultrasound':
                        dp.do_blur(x=x, y=y, file_prefix=file_prefix,
                                   filename=self._extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder, path_to_augment=self.dir_augmented,
                                   path_to_params=self.dir_params)

                    if self.do_deform:
                        for i in range(self.num_deform):
                            dp.do_deform(x=x, y=y, file_prefix=file_prefix,
                                         filename=self._extract_name_from_path(filename, without_ch=False),
                                         end_folder=end_folder,
                                         path_to_augment=self.dir_augmented, path_to_params=self.dir_params,
                                         num_deform=i)

                    if self.do_crop:
                        dp.do_crop(x=x, y=y, file_prefix=file_prefix,
                                   filename=self._extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder, path_to_augment=self.dir_augmented,
                                   path_to_params=self.dir_params)

                # additionally to the processed_all files the flipped ones are done for some augmentations
                flipped_to_be_aug = dp.ret_all_files_in_folder(folder_path=self.dir_processed + '/augmented/flip/' +
                                                               end_folder, full_names=True)
                if self.do_heavy_augment:
                    r_channels_to_be_aug = dp.ret_all_files_in_folder(folder_path=self.dir_processed +
                                                                      '/augmented/rchannels/' + end_folder,
                                                                      full_names=True)
                    flipped_to_be_aug = flipped_to_be_aug + r_channels_to_be_aug
                for filename in flipped_to_be_aug:
                    print('augmenting file', self._extract_name_from_path(filename, without_ch=False))
                    x = self._load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_low')
                    y = self._load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_high')

                    if self.do_deform:
                        for i in range(self.num_deform):
                            dp.do_deform(x=x, y=y, file_prefix=file_prefix,
                                         filename=self._extract_name_from_path(filename, without_ch=False),
                                         end_folder=end_folder,
                                         path_to_augment=self.dir_augmented, path_to_params=self.dir_params,
                                         num_deform=i)

                    if self.do_blur and end_folder == 'ultrasound':
                        dp.do_blur(x=x, y=y, file_prefix=file_prefix,
                                   filename=self._extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder, path_to_augment=self.dir_augmented,
                                   path_to_params=self.dir_params)

                    if self.do_crop:
                        dp.do_crop(x=x, y=y, file_prefix=file_prefix,
                                   filename=self._extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder,
                                   path_to_augment=self.dir_augmented, path_to_params=self.dir_params)

        elif self.data_type == self.accepted_data_types[1]:
            print('The new data set is to be augmented here. Not done yet.')
        else:
            print('This should be an empty else, to be stopped before coming here.')

    def _clear_aug_directories(self):
        # delete all files from the chosen augmented directories
        for aug_dir in self.names_of_augment:
            for im_dir in ['ultrasound', 'optoacoustic']:
                if im_dir == 'ultrasound':
                    file_prefix = 'US'
                else:
                    file_prefix = 'OA'
                if self.pro_and_augm_only_image_type and not self.image_type == file_prefix:
                    continue
                path_to_dir = self.dir_augmented + '/' + aug_dir + '/' + im_dir
                filelist = os.listdir(path_to_dir)
                for f in filelist:
                    os.remove(os.path.join(path_to_dir, f))

    def update_mean_var(self, prev_agg, data):
        (n_prev_obs, prev_mean, prev_var) = prev_agg
        m = n_prev_obs
        n = 1
        for dim_shape in data.shape:
            n = n * dim_shape

        newmean = data.mean()
        newvar = data.var()
        n_obs = m + n

        mean = m / (n_obs) * prev_mean + n / (n_obs) * newmean
        var = m / (n_obs) * prev_var + n / (n_obs) * newvar + m * n / (n_obs) ** 2 * (prev_mean - newmean) ** 2

        return (n_obs, mean, var)

    def _get_scale_center(self):
        print('Calculates scaling parameters')
        # Initialize values
        if self.image_type == 'US':
            i = 0
            temp = 1
            shape_for_mean_image = [401, 401]
            max_data_high = -np.inf * temp
            min_data_high = np.inf * temp
            max_data_low = -np.inf * temp
            min_data_low = np.inf * temp
            sum_image_high = np.zeros(shape_for_mean_image)
            sum_image_low = np.zeros(shape_for_mean_image)
            # do loop:
            for file in self.train_file_names:
                image_sign = self.image_type + '_high'
                # get high image
                image = self._load_file_to_numpy(file, image_sign)
                # get maximums/minimums of the image
                maxs = np.max(image)
                mins = np.min(image)
                # update maximums
                max_data_high = np.maximum(maxs, max_data_high)
                min_data_high = np.minimum(mins, min_data_high)
                # add image to sum (for mean)
                if self.image_type == 'US':
                    sum_image_high += image
                else:
                    sum_image_high += np.mean(image, axis=2)

                image_sign = self.image_type + '_low'
                # get low image
                image = self._load_file_to_numpy(file, image_sign)
                maxs = np.max(image)
                mins = np.min(image)
                # update maximums
                max_data_low = np.maximum(maxs, max_data_low)
                min_data_low = np.minimum(mins, min_data_low)
                # add image to sum (for mean)
                if self.image_type == 'US':
                    sum_image_low += image
                else:
                    sum_image_low += np.mean(image, axis=2)
                # increase counter
                i += 1
            # calculate mean images
            mean_image_high = sum_image_high / i
            mean_image_low = sum_image_low / i
        else:
            low_mean = 0
            low_var = 0
            count_low = 0
            high_mean = 0
            high_var = 0
            count_high = 0
            for file in self.train_file_names:
                image_sign = self.image_type + '_high'
                image_high = self._load_file_to_numpy(file, image_sign)
                image_sign = self.image_type + '_low'
                image_low = self._load_file_to_numpy(file, image_sign)
                # truncating images
                lq = np.quantile(image_low, self.trunc_points)
                hq = np.quantile(image_high, self.trunc_points)
                trunc_low = image_low.clip(lq[0], lq[1])
                trunc_high = image_high.clip(hq[0], hq[1])
                # update values
                low_aggr = self.update_mean_var((count_low, low_mean, low_var), trunc_low)
                (count_low, low_mean, low_var) = low_aggr
                high_aggr = self.update_mean_var((count_high, high_mean, high_var), trunc_high)
                (count_high, high_mean, high_var) = high_aggr

        # construct dictionaries and save parameters
        if self.image_type == 'US':
            us_scale_params = {'US_low': [min_data_low, max_data_low], 'US_high': [min_data_high, max_data_high]}
            us_mean_images = {'US_low': mean_image_low, 'US_high': mean_image_high}
            with open(self.dir_params + '/scale_and_center' + '/US_scale_params', 'wb') as handle:
                pickle.dump(us_scale_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.dir_params + '/scale_and_center' + '/US_mean_images', 'wb') as handle:
                pickle.dump(us_mean_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            oa_scale_params = {'OA_low': low_var, 'OA_high': high_var, 'trunc_points': self.trunc_points}
            oa_mean_images = {'OA_low': low_mean, 'OA_high': high_mean, 'trunc_points': self.trunc_points}
            with open(self.dir_params + '/scale_and_center' + '/OA_scale_params', 'wb') as handle:
                pickle.dump(oa_scale_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.dir_params + '/scale_and_center' + '/OA_mean_images', 'wb') as handle:
                pickle.dump(oa_mean_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

        ##################################################################
        # ###### Data Normalization ######################################
        ##################################################################
    def scale_image(self, image, min_data, max_data, min_out=-1, max_out=1):
        """ scales the input image from [min_data,max_data] to [min_out, max_out]
            input: image: for US (H,W) array, for OA: (H,W,C) array
                   min_data, max_data: minimum and maximum over the data set (channel by channel)
                        for US: floats, for OA: arrays of shape (C,)
                   image_type: 'US' or 'OA'
            output: image_out: array with same shape as image
        """

        factor = (max_out - min_out) / (max_data - min_data)
        additive = min_out - min_data * (max_out - min_out) / (max_data - min_data)
        image_out = image * factor + additive
        return image_out

    def scale_batch(self, batch, min_data, max_data, min_out=-1, max_out=1):
        """ scales the input batch from [min_data,max_data] to [min_out, max_out]
                input: batch: for US (N,H,W) array, for OA: (N,H,W,C) array
                       min_data, max_data: minimum and maximum over the data set (channel by channel)
                            for US: floats, for OA: arrays of shape (C,)
                       image_type: 'US' or 'OA'
                output: batch_out array with same shape as batch
        """
        factor = (max_out - min_out) / (max_data - min_data)
        additive = min_out - min_data * (max_out - min_out) / (max_data - min_data)
        batch_out = batch * factor + additive
        return batch_out

    def scale_and_center(self, batch, scale_params, mean_image):
        """ scales and centers the input batch given the scale_params and the mean_image
                input: batch: for US (N,H,W) array, for OA: (N,H,W,C) array
                       scale_params: minimum and maximum over the data set (channel by channel)
                            for US: array of shape (2,), for OA: arrays of shape (2,C)
                        mean_image: mean image for the whole data set
                            for US: array of shape (H,W), for OA: (H,W,C)
                       image_type: 'US' or 'OA'
                output: batch_out array with same shape as batch"""
        if self.image_type == 'US':
            [min_data, max_data] = scale_params
            mean_scaled = self.scale_image(image=mean_image, min_data=min_data, max_data=max_data)
            batch_out = self.scale_batch(batch=batch, min_data=min_data, max_data=max_data)
            batch_out = batch_out - mean_scaled
        else:
            batch_out = (batch - mean_image)/np.sqrt(scale_params)
        return batch_out

    def scale_and_center_reverse(self, batch, scale_params, mean_image):
        """ undoes the scaling and centering of the input batch given the scale_params and the mean_image
                    input: batch: for US (N,H,W) array, for OA: (N,H,W,C) array
                           scale_params: minimum and maximum over the data set (channel by channel)
                                for US: array of shape (2,), for OA: arrays of shape (2,C)
                            mean_image: mean image for the whole data set
                                for US: array of shape (H,W), for OA: (H,W,C)
                           image_type: 'US' or 'OA'
                    output: batch_out array with same shape as batch"""
        if self.image_type == 'US':
            [min_data, max_data] = scale_params
            mean_scaled = self.scale_image(image=mean_image, min_data=min_data, max_data=max_data)
            batch_out = batch + mean_scaled
            batch_out = self.scale_batch(batch_out, min_data=-1, max_data=1, min_out=min_data,
                                         max_out=max_data)
        else:
            batch_out = batch*np.sqrt(scale_params) + mean_image

        return batch_out
    def load_params(self, param_type, dir_params=None):
        """ loads the specified parameters from file
            input: image_type: 'US' or 'OA'
                    param_type: 'scale' or 'mean_image' (maybe more options later)
            output: params_low, params_high: the parameters."""
        # dir_params = '/mnt/local/mounted'
        if not self.do_scale_center:
            print('No Scaling done due to parameter do_scale_center=False.')
            return None, None
        if dir_params is None:
            dir_params = self.dir_params + '/' + 'scale_and_center/'
        if param_type in ['scale_params', 'mean_images']:
            file_name = self.image_type + '_' + param_type
            filepath = dir_params + file_name
            with open(filepath, 'rb') as handle:
                params = pickle.load(handle)
        else:
            sys.exit('invalid parameter type')
        dic_key_low = self.image_type + '_low'
        dic_key_high = self.image_type + '_high'
        params_low = params[dic_key_low]
        params_high = params[dic_key_high]
        if self.image_type == 'OA':
            params_trunc_points = params['trunc_points']
            if self.trunc_points != params_trunc_points:
                sys.exit('Truncation of saved parameters does not fit the chosen truncation. Truncation of saved parameters is ' + str(params_trunc_points))
        return params_low, params_high

    ##################################################################
    # ###### Torch tensor shaping ####################################
    ##################################################################

    def truncate_images_in_batch(self, batch, trunc_points):
        n_images = batch.shape[0]
        for im_ind in range(n_images):
            image = batch[im_ind, :, :, :]
            quantiles = np.quantile(image, trunc_points)
            batch[im_ind, :, :, :] = image.clip(quantiles[0], quantiles[1])
        return batch

    def scale_and_parse_to_tensor(self, batch_files, scale_params_low, scale_params_high,
                                  mean_image_low, mean_image_high):

        x, y = self.create_train_batches(batch_files)
        if self.image_type == 'OA':
                x = self.truncate_images_in_batch(x, self.trunc_points)
                y = self.truncate_images_in_batch(y, self.trunc_points)

        if self.do_scale_center:
            scale_center_x_val = self.scale_and_center(x, scale_params_low, mean_image_low)
            scale_center_y_val = self.scale_and_center(y, scale_params_high, mean_image_high)
        else:
            scale_center_x_val = x
            scale_center_y_val = y

        if self.image_type == 'US' and self.data_type == 'homo':
            scale_center_x_val = np.array([scale_center_x_val])
            scale_center_y_val = np.array([scale_center_y_val])

            # (C, N, H, W) to (N, C, H, W)
            scale_center_x_val = scale_center_x_val.reshape(scale_center_x_val.shape[1], scale_center_x_val.shape[0],
                                                            scale_center_x_val.shape[2], scale_center_x_val.shape[3])
            scale_center_y_val = scale_center_y_val.reshape(scale_center_y_val.shape[1], scale_center_y_val.shape[0],
                                                            scale_center_y_val.shape[2], scale_center_y_val.shape[3])

        else:
            # (N, H, W, C) to (N, C, H, W)
            scale_center_x_val = np.moveaxis(scale_center_x_val, [0, 1, 2, 3], [0, 2, 3, 1])
            scale_center_y_val = np.moveaxis(scale_center_y_val, [0, 1, 2, 3], [0, 2, 3, 1])

        input_tensor, target_tensor = torch.from_numpy(scale_center_x_val), torch.from_numpy(scale_center_y_val)

        return input_tensor, target_tensor
