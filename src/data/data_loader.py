import os
import pickle
import sys
from pathlib import Path
import numpy as np
import scipy.io
import random
import data.augmentation
import torch


class ProcessData(object):
    """
    TODO: Description of data loader in general; keep it growing
    train_ratio:    Gives the train and validation split for the model
    process_raw:    Process the raw data in the input folder and load them into processed folder
    image_type:     Either 'US' or 'OA' to select which data should be loaded
    do_augment:     Do the augmentation and save them in the corresponding folders
    """

    def __init__(self,
                 train_ratio,
                 image_type,
                 single_sample=False,
                 add_augment=True,
                 do_augment=False,
                 process_raw_data=False,
                 do_flip=True,
                 do_deform=True,
                 num_deform = 3,
                 do_blur=True,
                 do_crop=True,
                 get_scale_center=True):

        # initialize and write into self, then call the prepare data and return the data to the trainer
        self.train_ratio = train_ratio
        self.do_augment = do_augment  # call data_augment method
        self.do_flip = do_flip
        self.do_deform = do_deform
        self.num_deform = num_deform
        self.do_blur = do_blur
        self.do_crop = do_crop

        self.add_augment = add_augment  # bool if augmented data should be included in training

        self.image_type = image_type
        self.single_sample = single_sample  # if this is True only a single image will be loaded in the batch (dev)

        self.project_root_dir = str(Path().resolve().parents[1])  # root directory
        # self.project_root_dir = '/mnt/local/mounted'

        self.dir_raw_in = self.project_root_dir + '/data' + '/raw' + '/new_in'
        self.dir_processed_all = self.project_root_dir + '/data' + '/processed' + '/processed_all'
        self.dir_processed = self.project_root_dir + '/data' + '/processed'
        self.dir_augmented = self.project_root_dir + '/data' + '/processed' + '/augmented'
        self.all_folder = False  # check if raw folder was already processed
        self.process_oa = True  # process raw oa data
        self.process_us = True  # process raw us data
        self.augment_oa = True  # augment processed oa data
        self.augment_us = True  # augment processed us data
        self.process_raw = process_raw_data  # call method _process_raw_data
        self.get_scale_center = get_scale_center  # get scaling and mean image and store them
        self.dir_params = self.project_root_dir + '/params'
        self.set_random_seed = 42  # set a random seed to enable reproducable samples

        # run _prepare_data which calls the methods for preparartion, also augmentation etc.
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
            self.train_file_names, self.val_file_names = self._train_val_split(original_file_names=self.original_file_names)
            self._add_augmented_file_names_to_train()
            self.train_file_names = self._delete_val_from_augmented(val_names=self.val_file_names,
                                                               train_names=self.train_file_names)
        if self.get_scale_center:
            self._get_scale_center()

        self.X_val, self.Y_val = self._load_processed_data(full_file_names=self.val_file_names)

    ##################################################################
    ####### Data Loading and Preparation #############################
    ##################################################################

    def batch_names(self, batch_size):
        # shuffle the train_file_names; this gets called every epoch
        self.set_random_seed = self.set_random_seed + 1
        random.seed(self.set_random_seed)
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
        in_directories = [s for s in os.listdir(self.dir_raw_in) if '.' not in s]
        print("Preprocess raw data")
        if not self.all_folder:
            skip_dirs = []
            for sub in in_directories:
                if (next((True for s in os.listdir(self.dir_processed_all + '/ultrasound') if sub in s), False)
                    and next((True for s in os.listdir(self.dir_processed_all + '/optoacoustic') if sub in s), False)):
                    skip_dirs.append(sub)
                    print("As preprocessed data already exist, skip Folder:" + sub)
            # skip already processed folders
            in_directories = list(set(skip_dirs) ^ set(in_directories))

        for chunk_folder in in_directories:
            sample_directories = [s for s in os.listdir(self.dir_raw_in + '/' + chunk_folder) if '.' not in s]
            print("Processing data from raw input folder: " + chunk_folder)

            for sample_folder in sample_directories:
                in_files = os.listdir(self.dir_raw_in + '/' + chunk_folder + '/' + sample_folder)
                us_file = [s for s in in_files if 'US_' in s]
                oa_file = [s for s in in_files if 'OA_' in s]
                if us_file and self.process_us:
                    us_raw = scipy.io.loadmat(self.dir_raw_in + '/' + chunk_folder + '/' +
                                              sample_folder + '/' + us_file[0])
                    for i in range(us_raw['US_low'].shape[2]):
                        name_us_low = 'US_low_' + chunk_folder + '_' + sample_folder + '_ch' + str(i)
                        name_us_high = 'US_high_' + chunk_folder + '_' + sample_folder + '_ch' + str(i)  #
                        name_us_save = 'US_' + chunk_folder + '_' + sample_folder + '_ch' + str(i)
                        dict_us_single = {name_us_low: us_raw['US_low'][:, :, i],
                                          name_us_high: us_raw['US_high'][:, :, i]}
                        self._save_dict_with_pickle(file=dict_us_single,
                                                    folder_name='processed_all/ultrasound', file_name=name_us_save)

                if oa_file and self.process_oa:
                    oa_raw = scipy.io.loadmat(self.dir_raw_in + '/' + chunk_folder + '/' +
                                              sample_folder + '/' + oa_file[0])

                    name_oa_low = 'OA_low_' + chunk_folder + '_' + sample_folder
                    name_oa_high = 'OA_high_' + chunk_folder + '_' + sample_folder
                    name_oa_save = 'OA_' + chunk_folder + '_' + sample_folder
                    dict_oa = {name_oa_low: oa_raw['OA_low'],
                               name_oa_high: oa_raw['OA_high']}
                    self._save_dict_with_pickle(file=dict_oa, folder_name='processed_all/optoacoustic',
                                                file_name=name_oa_save)

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
            sys.exit("Error: No valid image_type selected!")
        else:
            if self.image_type == 'US':
                end_folder = 'ultrasound'
            else:
                end_folder = 'optoacoustic'
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
            if self.do_blur:
                self.train_file_names = self._names_to_list(folder_name=path_augmented + '/blur' + '/' + end_folder,
                                                            name_list=self.train_file_names)
            if self.do_deform:
                self.train_file_names = self._names_to_list(folder_name=path_augmented + '/deform' + '/' + end_folder,
                                                            name_list=self.train_file_names)
            if self.do_flip:
                self.train_file_names = self._names_to_list(folder_name=path_augmented + '/flip' + '/' + end_folder,
                                                            name_list=self.train_file_names)

    def _delete_val_from_augmented(self, val_names, train_names):
        # deletes the augmented data from the validation set from the training files

        names = [s for s in train_names if not self._detect_val_in_augment(s, val_names)]

        return names

    def _detect_val_in_augment(self, string, val_list):

        contained_in_val = any(self._extract_name_from_path(name) in string for name in val_list)

        return contained_in_val

    def _extract_name_from_path(self,string):
        # a small helper function to get the file name from the whole path
        # needed because we can't use os.path on server
        filename = ''
        found_slash = True
        for i in reversed(range(len(string))):
            sub = string[i]
            if sub == '/':
                found_slash = False
            if found_slash:
                filename = sub + filename
        return filename

    def _names_to_list(self, folder_name, name_list):
        # extract file names from folder and add path name to it
        file_names = [s for s in os.listdir(folder_name) if '.DS_' not in s]
        # add path to file names and add them to list
        name_list.extend([str(folder_name) + '/' + s for s in file_names])
        return name_list

    def _load_processed_data(self, full_file_names):
        # load the already preprocessed data and store it into X (low) and Y (high) numpy array
        # full_file_names:  iterable list of complete names: so path/filename

        X = np.array(
            [np.array(self._load_file_to_numpy(full_file_name=fname,
                                               image_sign=self.image_type + '_low')) for fname in full_file_names])
        Y = np.array(
            [np.array(self._load_file_to_numpy(full_file_name=fname,
                                               image_sign=self.image_type + '_high')) for fname in full_file_names])

        return X, Y

    def _load_file_to_numpy(self, full_file_name, image_sign):
        # helper function to load and read the data; pretty inefficient right now
        #  as we need to open every dict two times
        with open(full_file_name, 'rb') as handle:
            sample = pickle.load(handle)
        sample_array = [value for key, value in sample.items() if image_sign in key][0]
        return sample_array

    def _save_dict_with_pickle(self, file, folder_name, file_name):
        # use this to save pairs of low and high quality pictures
        with open(self.dir_processed + '/' + folder_name + '/' + file_name, 'wb') as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_params_with_pickle(self, file, folder_name, file_name):
        # use this to save pairs of low and high quality pictures
        with open(self.dir_params + '/' + folder_name + '/' + file_name, 'wb') as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##################################################################
    ####### Data Augmentation ########################################
    ##################################################################

    def _augment_data(self):
        # set random seed
        random.seed(self.set_random_seed)
        np.random.seed(self.set_random_seed)

        # load the raw data in .mat format, split up the us and oa and load them in dictionaries into processed folder
        proc_directories = [s for s in os.listdir(self.dir_processed_all) if '.' not in s]
        if self.image_type == 'OA' and self.do_blur:
            print('No blur augmentation for OA data')

        for chunk_folder in proc_directories:

            aug_files = os.listdir(self.dir_processed_all + '/' + chunk_folder)

            us_file = [s for s in aug_files if 'US_' in s]
            oa_file = [s for s in aug_files if 'OA_' in s]

            if us_file and self.image_type == 'US':

                for file in us_file:
                    X = self._load_file_to_numpy(full_file_name=str(self.dir_processed_all+ "/ultrasound") + "/" + file,
                                                 image_sign=self.image_type+'_low')
                    Y = self._load_file_to_numpy(full_file_name=str(self.dir_processed_all + "/ultrasound") + "/" + file,
                                                 image_sign=self.image_type+'_high')
                    print("augmenting file", file)
                    if self.do_blur:
                        aug_X, aug_Y, params = data.augmentation.blur(X, Y,
                                                              lower_lim=1, upper_lim=3)

                        name_oa_low = 'US_low_ultrasound_blur'
                        name_oa_high = 'US_high_ultrasound_blur'
                        name_oa_save = 'US_ultrasound_' + file + "_blur"
                        
                        dict_oa = {name_oa_low: aug_X,
                                   name_oa_high: aug_Y}
                        self._save_dict_with_pickle(dict_oa, "augmented/blur/ultrasound", name_oa_save)
                        name_oa_save_params = name_oa_save + '_params'
                        self._save_params_with_pickle(params, 'augmentation/blur/ultrasound', name_oa_save_params)

                    if self.do_deform:
                        for i in range(self.num_deform):
                            aug_X, aug_Y, params = data.augmentation.elastic_deform(X, Y)

                            name_oa_low = 'US_low_ultrasound_deform'
                            name_oa_high = 'US_high_ultrasound_deform'
                            name_oa_save = 'US_ultrasound_' + file + "_deform_" + str(i)

                            dict_oa = {name_oa_low: aug_X,
                                        name_oa_high: aug_Y}
                            self._save_dict_with_pickle(dict_oa, "augmented/deform/ultrasound", name_oa_save)
                            name_oa_save_params = name_oa_save + '_params'
                            self._save_params_with_pickle(params, 'augmentation/deform/ultrasound', name_oa_save_params)

                    if self.do_crop:
                        aug_X, aug_Y, params = data.augmentation.crop_stretch(X, Y)
                        name_oa_low = 'US_low_ultrasound_crop'
                        name_oa_high = 'US_high_ultrasound_crop'
                        name_oa_save = 'US_ultrasound_' + file + "_crop"

                        dict_oa = {name_oa_low: aug_X,
                                   name_oa_high: aug_Y}
                        self._save_dict_with_pickle(dict_oa, "augmented/crop/ultrasound", name_oa_save)
                        name_oa_save_params = name_oa_save + '_params'
                        self._save_params_with_pickle(params, 'augmentation/crop/ultrasound', name_oa_save_params)

                    if self.do_flip:
                        aug_X, aug_Y = data.augmentation.flip(X, Y)
                        name_oa_low = 'US_low_ultrasound_flip'
                        name_oa_high = 'US_high_ultrasound_flip'
                        name_oa_save = 'US_ultrasound_' + file + "_flip"

                        dict_oa = {name_oa_low: aug_X,
                                   name_oa_high: aug_Y}
                        self._save_dict_with_pickle(dict_oa, "augmented/flip/ultrasound", name_oa_save)

            if oa_file and self.image_type == 'OA':

                for file in oa_file:
                    X = self._load_file_to_numpy(full_file_name= str(self.dir_processed_all + "/optoacoustic") + "/" + file,
                                                 image_sign=self.image_type + '_low')
                    Y = self._load_file_to_numpy(full_file_name=str(self.dir_processed_all + "/optoacoustic") + "/" + file,
                                                 image_sign=self.image_type + '_high')
                    print("augmenting file", file)

                    if self.do_blur:
                        pass
                        #### we don't blur for optoacoustic
                        #aug_X, aug_Y, params = data.augmentation.blur(X, Y, lower_lim=1, upper_lim=3)
                        #name_oa_low = 'OA_low_optocoustic_blur'
                        #name_oa_high = 'OA_high_optoacoustic_blur'
                        #name_oa_save = 'OA_optoacoustic_' + file + "_blur"
                        #dict_oa = {name_oa_low: aug_X,
                        #           name_oa_high: aug_Y}
                        #self._save_dict_with_pickle(dict_oa, "augmented/blur/optoacoustic", name_oa_save)
                        #name_oa_save_params = name_oa_save + '_params'
                        #self._save_params_with_pickle(params, 'augmentation/blur/optoacoustic', name_oa_save_params)
                    if self.do_deform:
                        for i in range(self.num_deform):
                            aug_X, aug_Y, params = data.augmentation.elastic_deform(X, Y)
                            name_oa_low = 'OA_low_optoacoustic_deform'
                            name_oa_high = 'OA_high_optoacoustic_deform'
                            name_oa_save = 'OA_optoacoustic_' + file + "_deform_" + str(i)
                            dict_oa = {name_oa_low: aug_X,
                                       name_oa_high: aug_Y}
                            self._save_dict_with_pickle(dict_oa, "augmented/deform/optoacoustic", name_oa_save)
                            name_oa_save_params = name_oa_save + '_params'
                            self._save_params_with_pickle(params, 'augmentation/deform/optoacoustic', name_oa_save_params)

                    if self.do_crop:
                        aug_X, aug_Y, params = data.augmentation.crop_stretch(X, Y)
                        name_oa_low = 'OA_low_optoacoustic_crop'
                        name_oa_high = 'OA_high_optoacoustic_crop'
                        name_oa_save = 'OA_optoacoustic_' + file + "_crop"

                        dict_oa = {name_oa_low: aug_X,
                                   name_oa_high: aug_Y}
                        self._save_dict_with_pickle(dict_oa, "augmented/crop/optoacoustic", name_oa_save)
                        name_oa_save_params = name_oa_save + '_params'
                        self._save_params_with_pickle(params, 'augmentation/crop/optoacoustic', name_oa_save_params)

                    if self.do_flip:
                        aug_X, aug_Y = data.augmentation.flip(X, Y)
                        name_oa_low = 'OA_low_optoacoustic_flip'
                        name_oa_high = 'OA_high_optoacoustic_flip'
                        name_oa_save = 'OA_optoacoustic_' + file + "_flip"

                        dict_oa = {name_oa_low: aug_X,
                                   name_oa_high: aug_Y}
                        self._save_dict_with_pickle(dict_oa, "augmented/flip/optoacoustic", name_oa_save)

    def _get_scale_center(self):
        # Initialize values
        i = 0
        if self.image_type == 'US':
            temp = 1
            shape_for_mean_image = [401, 401]
        else:
            temp = np.ones(28)
            shape_for_mean_image = [401, 401, 28]
        max_data_high = -np.inf*temp
        min_data_high = np.inf*temp
        max_data_low = -np.inf*temp
        min_data_low = np.inf*temp
        sum_image_high = np.zeros(shape_for_mean_image)
        sum_image_low = np.zeros(shape_for_mean_image)
        # do loop:
        for file in self.train_file_names:
            image_sign = self.image_type + '_high'
            # get high image
            image = self._load_file_to_numpy(file, image_sign)
            # get maximums/minimums of the image
            if self.image_type == 'US':
                maxs = np.max(image)
                mins = np.min(image)
            else:
                maxs = np.amax(image, axis=(0, 1))
                mins = np.amin(image, axis=(0, 1))
            # update maximums
            max_data_high = np.maximum(maxs, max_data_high)
            min_data_high = np.minimum(mins, min_data_high)
            # add image to sum (for mean)
            sum_image_high += image

            image_sign = self.image_type + '_low'
            # get low image
            image = self._load_file_to_numpy(file, image_sign)
            if self.image_type == 'US':
                maxs = np.max(image)
                mins = np.min(image)
            else:
                maxs = np.amax(image, axis=(0, 1))
                mins = np.amin(image, axis=(0, 1))
            # update maximums
            max_data_low = np.maximum(maxs, max_data_low)
            min_data_low = np.minimum(mins, min_data_low)
            # add image to sum (for mean)
            sum_image_low += image
            # increase counter
            i += 1
        # calculate mean images
        mean_image_high = sum_image_high / i
        mean_image_low = sum_image_low / i

        # construct dictionaries and save parameters
        if self.image_type == 'US':
            US_scale_params = {'US_low': [min_data_low, max_data_low], 'US_high': [min_data_high, max_data_high]}
            US_mean_images = {'US_low': mean_image_low, 'US_high': mean_image_high}
            with open(self.dir_params + '/scale_and_center' + '/US_scale_params', 'wb') as handle:
                pickle.dump(US_scale_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.dir_params + '/scale_and_center' + '/US_mean_images', 'wb') as handle:
                pickle.dump(US_mean_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            OA_scale_params = {'OA_low': [min_data_low, max_data_low], 'OA_high': [min_data_high, max_data_high]}
            OA_mean_images = {'OA_low': mean_image_low, 'OA_high': mean_image_high}
            # CAUTION: the OA mean image gets stored in the (C,N,N) shape!!!! (that's what the moveaxis is doing)
            with open(self.dir_params + '/scale_and_center' + '/OA_scale_params', 'wb') as handle:
                pickle.dump(OA_scale_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.dir_params + '/scale_and_center' + '/OA_mean_images', 'wb') as handle:
                pickle.dump(OA_mean_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

        ##################################################################
        ####### Data Normalization #######################################
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
        if self.image_type == 'US':
            image_out = image * factor + additive
        else:
            image_out = image * factor[None, None, :] + additive[None, None, :]
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
        if self.image_type == 'US':
            batch_out = batch * factor + additive
        else:
            batch_out = batch * factor[None, None, None, :] + additive[None, None, None, :]
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
        [min_data, max_data] = scale_params
        mean_scaled = self.scale_image(image=mean_image, min_data=min_data, max_data=max_data)
        batch_out = self.scale_batch(batch=batch, min_data=min_data, max_data=max_data)
        batch_out = batch_out - mean_scaled
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
        [min_data, max_data] = scale_params
        mean_scaled = self.scale_image(image=mean_image, min_data=min_data, max_data=max_data)
        batch_out = batch + mean_scaled
        batch_out = self.scale_batch(batch_out, min_data=-1, max_data=1, min_out=min_data,
                                max_out=max_data)
        # just for not using the mean addition
        # batch_out = scale_batch(batch, min_data=-1, max_data=1, image_type=image_type, min_out=min_data,
        #                        max_out = max_data)
        return batch_out

    def load_params(self, param_type):
        """ loads the specified parameters from file
            input: image_type: 'US' or 'OA'
                    param_type: 'scale' or 'mean_image' (maybe more options later)
            output: params_low, params_high: the parameters."""
        # dir_params = '/mnt/local/mounted'
        dir_params = self.project_root_dir + '/' + 'params'
        if param_type in ['scale_params', 'mean_images']:
            file_name = self.image_type + '_' + param_type
            filepath = dir_params + '/' + 'scale_and_center' + '/' + file_name
            with open(filepath, 'rb') as handle:
                params = pickle.load(handle)
        else:
            print('invalid parameter type')
        dic_key_low = self.image_type + '_low'
        dic_key_high = self.image_type + '_high'
        params_low = params[dic_key_low]
        params_high = params[dic_key_high]
        return params_low, params_high

    ##################################################################
    ####### Torch tensor shaping #####################################
    ##################################################################

    def scale_and_parse_to_tensor(self, batch_files, scale_params_low, scale_params_high,
                            mean_image_low, mean_image_high):
        x, y = self.create_train_batches(batch_files)

        scale_center_x_val = self.scale_and_center(x, scale_params_low, mean_image_low)
        scale_center_y_val = self.scale_and_center(y, scale_params_high, mean_image_high)
        scale_center_x_val = np.array([scale_center_x_val])
        scale_center_y_val = np.array([scale_center_y_val])

        # (C, N, H, W) to (N, C, H, W)
        scale_center_x_val = scale_center_x_val.reshape(scale_center_x_val.shape[1], scale_center_x_val.shape[0],
                                                        scale_center_x_val.shape[2], scale_center_x_val.shape[3])
        scale_center_y_val = scale_center_y_val.reshape(scale_center_y_val.shape[1], scale_center_y_val.shape[0],
                                                        scale_center_y_val.shape[2], scale_center_y_val.shape[3])

        input_tensor, target_tensor = torch.from_numpy(scale_center_x_val), torch.from_numpy(scale_center_y_val)

        return input_tensor, target_tensor