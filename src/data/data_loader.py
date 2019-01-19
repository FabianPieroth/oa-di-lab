import os
import pickle
import sys
from pathlib import Path
import numpy as np
import random
import data.data_processing as dp
import torch
import scipy.io
from sklearn.decomposition import IncrementalPCA
from logger.oa_spectra_analysis.oa_for_DILab import linear_unmixing as lu


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
                 process_all_raw_folders=False,
                 process_raw_test=True,
                 pro_and_augm_only_image_type=False,
                 trunc_points_before_pca=(0.0001,0.9999),
                 oa_do_scale_center_before_pca=True,
                 oa_do_pca=False,
                 oa_pca_fit_ratio=1,
                 oa_pca_num_components=7,
                 pca_use_regress=False,
                 height_channel_oa=201,
                 use_regressed_oa=False,
                 add_f_test=False,
                 include_regression_error=False,
                 only_f_test_in_target=False,
                 channel_slice_oa=None,
                 do_flip=True,
                 do_deform=True,
                 num_deform=3,
                 do_blur=True,
                 do_crop=True,
                 do_rchannels=True,
                 num_rchannels=2,
                 do_speckle_noise=True,
                 get_scale_center=True,
                 do_scale_center=True,
                 hetero_mask_to_mask=False,
                 attention_mask='Not',
                 attention_anchors=None,
                 attention_input_dist=None,
                 attention_network_dist=None,
                 trunc_points=(0.0001, 0.9999),
                 logger_call=False):

        # initialize and write into self, then call the prepare data and return the data to the trainer
        self.train_ratio = train_ratio
        self.data_type = data_type
        self.accepted_data_types = ['homo', 'hetero', 'bi']

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

        self.num_val_images_hetero = 3

        self.do_speckle_noise = do_speckle_noise

        self.trunc_points_before_pca = trunc_points_before_pca
        self.oa_do_scale_center_before_pca = oa_do_scale_center_before_pca
        self.oa_do_pca = oa_do_pca
        self.oa_pca_fit_ratio = oa_pca_fit_ratio
        self.oa_pca_num_components = oa_pca_num_components
        self.pca_use_regress = pca_use_regress
        self.height_channel_oa = height_channel_oa
        self.use_regressed_oa = use_regressed_oa
        self.include_regression_error = include_regression_error
        self.add_f_test = add_f_test
        self.only_f_test_in_target = only_f_test_in_target

        self.attention_mask = attention_mask
        self.attention_anchors = attention_anchors
        self.attention_input_dist = attention_input_dist
        self.attention_network_dist = attention_network_dist

        # if self.attention_mask == 'complex':
        if self.attention_anchors is None:
            sys.exit('Please provide attention anchors for the distribution of the input or choose attention_mask' +
                     ' as simple.')
        if self.attention_input_dist is None:
            if len(self.attention_anchors) == 2:
                self.attention_input_dist = [len(self.attention_anchors) / 2, len(self.attention_anchors) / 2]
            else:
                sys.exit('Please provide a suitable attention input distribution.')
        if not len(self.attention_anchors) == np.sum(self.attention_input_dist):
            sys.exit('The chosen number of attention anchor points does not match the sum of the given attention' +
                     'input distribution')
        '''else:
            print('You have chosen the simple case, please make sure you have suitable attention_anchors')
            self.attention_input_dist = [1, 1]
            if self.attention_anchors is None:
                # TODO: Check the default here
                self.attention_anchors = [0.2, 1.0]'''

        if channel_slice_oa is None:
            self.channel_slice_oa = list(range(28))
        else:
            self.channel_slice_oa = channel_slice_oa

        self.hetero_mask_to_mask = hetero_mask_to_mask

        # the order of the following two lists has to be the same and has to be extended if new augmentations are done
        self.all_augment = ['flip', 'deform', 'blur', 'crop', 'rchannels', 'speckle_noise']
        self.bool_augment = [self.do_flip, self.do_deform, self.do_blur, self.do_crop, self.do_rchannels,
                             self.do_speckle_noise]
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
        self.process_all_raw_folders = process_all_raw_folders  # check if raw folder was already processed
        self.pro_and_augm_only_image_type = pro_and_augm_only_image_type
        self.process_oa = not self.pro_and_augm_only_image_type or \
                          (self.pro_and_augm_only_image_type and self.image_type == 'OA')  # process raw oa data
        self.process_us = not self.pro_and_augm_only_image_type or \
                          (self.pro_and_augm_only_image_type and self.image_type == 'US')  # process raw us data
        self.process_raw = process_raw_data  # call method _process_raw_data
        self.process_raw_test = process_raw_test  # call method _process_raw_test
        self.test_names = []
        self.get_scale_center = get_scale_center  # get scaling and mean image and store them
        self.do_scale_center = do_scale_center  # applies scale and center to the data
        self.trunc_points = trunc_points # quantiles at which to truncate OA data
        self.dir_params = self.project_root_dir + '/data' + '/' + self.data_type + '/params'
        self.set_random_seed = 42  # set a random seed to enable reproducable samples

        self.logger_call = logger_call  # initialise the data_loader but do nothing else

        if self.image_type == 'US':
            self.end_folder = 'ultrasound'
        else:
            self.end_folder = 'optoacoustic'

        # run _prepare_data which calls the methods for preparation, also augmentation etc.
        if not self.logger_call:
            self._prepare_data()

        # self.X_train, self.y_train, self.X_test, self.y_test, self.df_complete = self._prepare_data()

    def _prepare_data(self):
        # process data
        if self.process_raw:
            self._process_raw_data()

        if self.process_raw_test:
            self._process_raw_test()
            self.test_names = dp.ret_all_files_in_folder(self.dir_processed + '/test_set' + '/' + self.end_folder,
                                                         full_names=True)

        if self.do_augment:
            # TODO: check augmentations for case attention_mask='complex'
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

        if self.oa_do_pca:
            if self.process_raw == False or self.do_augment == False:
                sys.exit('Please process and augment the data before doing the pca')
            if self.image_type == 'US' or self.data_type == 'hetero':
                sys.exit('PCA not implemented for US or hetero data')
            # get normalization parameters before pca
            if self.oa_do_scale_center_before_pca:
                self._get_scale_center(before_pca=True)

            # fit PCA on targets in train set
            self.fit_pca_on_targets(self.train_file_names, self.oa_pca_fit_ratio, self.oa_pca_num_components)
            # join train, val and test file names
            self.all_data_files = []
            self.all_data_files.extend(self.train_file_names)
            self.all_data_files.extend(self.val_file_names)
            self.all_data_files.extend(self.test_names)
            # project all data files onto pca space and overwrite the files
            self.do_pca_and_save_data(self.all_data_files)

        if self.get_scale_center:
            self._get_scale_center()

        print('There are ' + str(len(self.train_file_names)) + ' files in the training set')

        print('There are ' + str(len(self.val_file_names)) + ' files in the validation set')

        self.X_val, self.Y_val = self.load_processed_data(full_file_names=self.val_file_names)

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
            batch_size = 1  # if only single sample is called, set batch_size on 1

        # give a list and return the corresponding batch names
        self.train_batch_chunks = np.array_split(np.array(self.train_file_names),
                                                 int(len(self.train_file_names) / batch_size))
        self.batch_number = len(self.train_batch_chunks)

    def create_train_batches(self, batch_names):
        # return the batches
        x, y = self.load_processed_data(full_file_names=batch_names)
        return x, y

    def _process_raw_data(self):
        # load the raw data in .mat format, split up the us and oa and load them in dictionaries into processed folder
        in_directories = dp.ret_all_files_in_folder(folder_path=self.dir_raw_in, full_names=False)
        print('Process raw data')
        if not self.process_all_raw_folders:
            skip_dirs = []
            for sub in in_directories:
                if (next((True for s in os.listdir(self.dir_processed_all + '/ultrasound') if sub in s), False)
                        and next((True for s in os.listdir(self.dir_processed_all + '/optoacoustic') if sub in s),
                                 False)):
                    skip_dirs.append(sub)
                    print('As preprocessed data already exist, skip Folder:' + sub)
            # skip already processed folders
            in_directories = list(set(skip_dirs) ^ set(in_directories))

        if self.process_all_raw_folders:
            # clear all files from processed_all folder, if all directories will be processed
            self._clear_dir(path_to_dir=self.dir_processed_all + '/' + self.end_folder)

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
                                       cut_half=True, height_channel=self.height_channel_oa,
                                       use_regressed_oa=self.use_regressed_oa,
                                       regression_coefficients=self._get_default_spectra(),
                                       include_regression_error=self.include_regression_error,
                                       add_f_test=self.add_f_test, only_f_test_in_target=self.only_f_test_in_target,
                                       channel_slice_oa=self.channel_slice_oa)

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
                                     save_folder=self.dir_processed_all, hetero_mask_to_mask=self.hetero_mask_to_mask,
                                     attention_mask=self.attention_mask, attention_input_dist=self.attention_input_dist)
                elif self.data_type == self.accepted_data_types[2]:
                    if not self.image_type == 'US':
                        sys.exit('There is only Ultrasound images in the bi data set.')
                    in_files = dp.ret_all_files_in_folder(folder_path=self.dir_raw_in + '/' +
                                                          chunk_folder + '/' + sample_folder,
                                                          full_names=False)
                    us_low_samples = [s for s in in_files if 'US_couplant' in s]
                    us_high_samples = [s for s in in_files if 'US_high' in s]
                    dp.pre_us_bi(new_in_folder=self.dir_raw_in, study_folder=chunk_folder, scan_num=sample_folder,
                                 filename_low=us_low_samples[0], filename_high=us_high_samples[0],
                                 save_folder=self.dir_processed_all, attention_mask=self.attention_mask,
                                 attention_anchors=self.attention_anchors,
                                 attention_input_dist=self.attention_input_dist)
                else:
                    print('This should be an empty else, to be stopped before coming here.')

    def _process_raw_test(self):
        # this is the pre-processing of the test set
        dir_test_set = self.project_root_dir + '/data' + '/' + self.data_type + '/raw' + '/test_set'
        in_directories = dp.ret_all_files_in_folder(folder_path=dir_test_set, full_names=False)
        save_dir = self.dir_processed + '/test_set'
        if len(in_directories) == 0:
            print('There are no files in the test set, the process step is therefore skipped.')
            return None
        # clear directory first so that there are not several channels from the same file
        self._clear_dir(path_to_dir=save_dir + '/' + self.end_folder)

        for chunk_folder in in_directories:
            sample_directories = dp.ret_all_files_in_folder(folder_path=dir_test_set + '/' + chunk_folder,
                                                            full_names=False)
            print('Processing data from test_set folder: ' + chunk_folder)
            for sample_folder in sample_directories:
                if self.data_type == self.accepted_data_types[0]:
                    in_files = dp.ret_all_files_in_folder(folder_path=dir_test_set + '/' +
                                                          chunk_folder + '/' + sample_folder,
                                                          full_names=False)
                    us_file = [s for s in in_files if 'US_' in s]
                    oa_file = [s for s in in_files if 'OA_' in s]

                    dp.pre_us_homo(new_in_folder=dir_test_set, study_folder=chunk_folder,
                                   filename=us_file[0], scan_num=sample_folder, save_folder=save_dir)

                    dp.pre_oa_homo(new_in_folder=dir_test_set, study_folder=chunk_folder,
                                   filename=oa_file[0], scan_num=sample_folder, save_folder=save_dir,
                                   cut_half=True, height_channel=self.height_channel_oa,
                                   use_regressed_oa=self.use_regressed_oa,
                                   regression_coefficients=self._get_default_spectra(),
                                   include_regression_error=self.include_regression_error,
                                   add_f_test=self.add_f_test, only_f_test_in_target=self.only_f_test_in_target,
                                   channel_slice_oa=self.channel_slice_oa)
                elif self.data_type == self.accepted_data_types[1]:
                    in_files = dp.ret_all_files_in_folder(folder_path=dir_test_set + '/' +
                                                                      chunk_folder + '/' + sample_folder,
                                                          full_names=False)
                    us_low_samples = [s for s in in_files if 'US_low' in s]
                    us_high_samples = [s for s in in_files if 'US_high' in s]
                    dp.pre_us_hetero(new_in_folder=dir_test_set, study_folder=chunk_folder, scan_num=sample_folder,
                                     filename_low=us_low_samples[0], filename_high=us_high_samples[0],
                                     save_folder=save_dir, hetero_mask_to_mask=self.hetero_mask_to_mask,
                                     attention_mask=self.attention_mask, attention_input_dist=self.attention_input_dist)
                elif self.data_type == self.accepted_data_types[2]:
                    if not self.image_type == 'US':
                        sys.exit('There is only Ultrasound images in the bi data set.')
                    in_files = dp.ret_all_files_in_folder(folder_path=dir_test_set + '/' +
                                                          chunk_folder + '/' + sample_folder,
                                                          full_names=False)
                    us_low_samples = [s for s in in_files if 'US_couplant' in s]
                    us_high_samples = [s for s in in_files if 'US_high' in s]
                    dp.pre_us_bi(new_in_folder=dir_test_set, study_folder=chunk_folder, scan_num=sample_folder,
                                 filename_low=us_low_samples[0], filename_high=us_high_samples[0],
                                 save_folder=save_dir, attention_mask=self.attention_mask,
                                 attention_anchors=self.attention_anchors,
                                 attention_input_dist=self.attention_input_dist)

    def _train_val_split(self, original_file_names):
        # this should only be called once at the beginning to ensure the same random seed
        random.seed(self.set_random_seed)
        if self.data_type == 'homo':
            original_file_names = random.sample(original_file_names, len(original_file_names))
            train_size = int(len(original_file_names) * self.train_ratio)
            self.train_names, val_names = original_file_names[:train_size], original_file_names[train_size:]
        elif self.data_type == 'hetero' or self.data_type == 'bi':
            shortened_file_names = list(set([self.extract_name_from_path(s) for s in original_file_names]))
            train_size = int(len(shortened_file_names) * self.train_ratio)
            short_train_names, short_val_names = shortened_file_names[:train_size], shortened_file_names[train_size:]
            self.train_names = self._retrieve_file_names_from_short(name_list_short=short_train_names,
                                                                    name_list_long=original_file_names)
            val_names = self._retrieve_file_names_from_short(name_list_short=short_val_names,
                                                             name_list_long=original_file_names,
                                                             num_images=self.num_val_images_hetero)
        else:
            print('You should not have come here. Check your work!')
        return self.train_names, val_names

    def _retrieve_original_file_names(self):
        # get all the complete file names (path/filename) of the selected data to train on
        if self.image_type not in ['US', 'OA']:
            sys.exit('Error: No valid image_type selected!')

        if len(dp.ret_all_files_in_folder(folder_path=self.dir_processed_all + '/' + self.end_folder,
                                          full_names=False)) == 0:
            sys.exit('There are no Files in the processed Folder, please run again with process_raw_data=True.')

        file_names = []
        # original images
        file_names = self._names_to_list(folder_name=self.dir_processed_all + '/' + self.end_folder,
                                         name_list=file_names)
        return file_names

    def _add_augmented_file_names_to_train(self):
        # add the file names of the augmented data to self.train_file_names
        path_augmented = self.dir_processed + '/augmented'

        if self.add_augment:
            for aug_name in self.names_of_augment:
                self.train_file_names = self._names_to_list(
                    folder_name=path_augmented + '/' + aug_name + '/' + self.end_folder,
                    name_list=self.train_file_names)

    def _retrieve_file_names_from_short(self, name_list_short, name_list_long, num_images=None):
        ret_list = []
        for short in name_list_short:
            if num_images is None:
                num_images = 9999
            path_names_list = [s for s in name_list_long if short in s]
            take_num_images = np.min([num_images, len(path_names_list)])
            ret_list = ret_list + list(np.random.permutation(path_names_list))[:take_num_images]

        return ret_list

    def _delete_val_from_augmented(self, val_names, train_names):
        # deletes the augmented data from the validation set from the training files
        names = [s for s in train_names if not self._detect_val_in_augment(s, val_names)]
        return names

    def _detect_val_in_augment(self, string, val_list):
        contained_in_val = any(self.extract_name_from_path(name) in string for name in val_list)
        return contained_in_val

    def extract_name_from_path(self, string, without_ch=True):
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

    def load_processed_data(self, full_file_names):
        # load the already preprocessed data and store it into X (low) and Y (high) numpy array
        # full_file_names:  iterable list of complete names: so path/filename
        x = np.array(
            [np.array(self.load_file_to_numpy(full_file_name=fname,
                                              image_sign=self.image_type + '_low')) for fname in full_file_names])
        y = np.array(
            [np.array(self.load_file_to_numpy(full_file_name=fname,
                                              image_sign=self.image_type + '_high')) for fname in full_file_names])

        return x, y

    def load_file_to_numpy(self, full_file_name, image_sign):
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

    # regression of optoacoustic images

    def _get_default_spectra(self):
        '''
        Function to get the default spectra used by functions in this module
        Default are clinical spectra w/o collagen

        # Returns:
            Clinical spectra of Hb, HbO2, Fat and Water.
            Shape is (4,28)
        '''
        spectra = scipy.io.loadmat(self.project_root_dir + '/src/' + 'logger/oa_spectra_analysis/clinical_spectra.mat')[
            'spectra_L2'].T
        return spectra[:4]

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
            if self.image_type == 'OA' and self.do_speckle_noise:
                print('No speckle noise augmentation for OA data')

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
                    print('augmenting file', self.extract_name_from_path(filename, without_ch=False))
                    x = self.load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_low')
                    y = self.load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_high')
                    if self.do_flip:
                        dp.do_flip(x=x, y=y, file_prefix=file_prefix,
                                   filename=self.extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder,
                                   path_to_augment=self.dir_augmented)

                    # the additional channels are only taken from the processed_all folder
                    if self.do_rchannels and end_folder == 'ultrasound':
                        dp.do_rchannels(end_folder=end_folder,
                                        filename=self.extract_name_from_path(filename, without_ch=False),
                                        read_in_folder=self.dir_raw_in,
                                        num_channels=self.num_rchannels, path_to_augment=self.dir_augmented)

                    if self.do_blur and end_folder == 'ultrasound':
                        dp.do_blur(x=x, y=y, file_prefix=file_prefix,
                                   filename=self.extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder, path_to_augment=self.dir_augmented,
                                   path_to_params=self.dir_params, data_type=self.data_type,
                                   attention_mask=self.attention_mask, attention_input_dist=self.attention_input_dist)

                    if self.do_deform and self.data_type == 'homo':
                        for i in range(self.num_deform):
                            dp.do_deform(x=x, y=y, file_prefix=file_prefix,
                                         filename=self.extract_name_from_path(filename, without_ch=False),
                                         end_folder=end_folder,
                                         path_to_augment=self.dir_augmented, path_to_params=self.dir_params,
                                         num_deform=i)

                    if self.do_crop:
                        dp.do_crop(x=x, y=y, file_prefix=file_prefix,
                                   filename=self.extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder, path_to_augment=self.dir_augmented,
                                   path_to_params=self.dir_params)

                    if self.do_speckle_noise and end_folder == 'ultrasound':
                        dp.do_speckle_noise(x=x, y=y, file_prefix=file_prefix,
                                            filename=self.extract_name_from_path(filename, without_ch=False),
                                            end_folder=end_folder, path_to_augment=self.dir_augmented,
                                            path_to_params=self.dir_params, data_type=self.data_type,
                                            attention_mask=self.attention_mask,
                                            attention_input_dist=self.attention_input_dist)

                # additionally to the processed_all files the flipped ones are done for some augmentations

                flipped_to_be_aug = dp.ret_all_files_in_folder(folder_path=self.dir_processed + '/augmented/flip/' +
                                                               end_folder, full_names=True)
                if self.do_heavy_augment:
                    r_channels_to_be_aug = dp.ret_all_files_in_folder(folder_path=self.dir_processed +
                                                                      '/augmented/rchannels/' + end_folder,
                                                                      full_names=True)
                    flipped_to_be_aug = flipped_to_be_aug + r_channels_to_be_aug
                for filename in flipped_to_be_aug:
                    print('augmenting file', self.extract_name_from_path(filename, without_ch=False))
                    x = self.load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_low')
                    y = self.load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_high')

                    if self.do_deform and self.data_type =='homo':
                        for i in range(self.num_deform):
                            dp.do_deform(x=x, y=y, file_prefix=file_prefix,
                                         filename=self.extract_name_from_path(filename, without_ch=False),
                                         end_folder=end_folder,
                                         path_to_augment=self.dir_augmented, path_to_params=self.dir_params,
                                         num_deform=i)

                    if self.do_blur and end_folder == 'ultrasound':
                        dp.do_blur(x=x, y=y, file_prefix=file_prefix,
                                   filename=self.extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder, path_to_augment=self.dir_augmented,
                                   path_to_params=self.dir_params, data_type=self.data_type,
                                   attention_mask=self.attention_mask, attention_input_dist=self.attention_input_dist)

                    if self.do_crop:
                        dp.do_crop(x=x, y=y, file_prefix=file_prefix,
                                   filename=self.extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder,
                                   path_to_augment=self.dir_augmented, path_to_params=self.dir_params)

                    if self.do_speckle_noise and end_folder == 'ultrasound':
                        dp.do_speckle_noise(x=x, y=y, file_prefix=file_prefix,
                                            filename=self.extract_name_from_path(filename, without_ch=False),
                                            end_folder=end_folder, path_to_augment=self.dir_augmented,
                                            path_to_params=self.dir_params, data_type=self.data_type,
                                            attention_mask=self.attention_mask,
                                            attention_input_dist=self.attention_input_dist)

        elif self.data_type == self.accepted_data_types[1]:
            if self.data_type == 'hetero' and self.do_deform:
                print('No deform augmentation for heterogeneous SoS data')
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

                for filename in to_be_aug_files:
                    print('augmenting file', self.extract_name_from_path(filename, without_ch=False))
                    x = self.load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_low')
                    y = self.load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_high')
                    if self.do_flip:
                        dp.do_flip(x=x, y=y, file_prefix=file_prefix,
                                   filename=self.extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder,
                                   path_to_augment=self.dir_augmented)
                    if self.do_blur and end_folder == 'ultrasound':
                        dp.do_blur(x=x, y=y, file_prefix=file_prefix,
                                   filename=self.extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder, path_to_augment=self.dir_augmented,
                                   path_to_params=self.dir_params, data_type=self.data_type,
                                   attention_mask=self.attention_mask, attention_input_dist=self.attention_input_dist)
                    if self.do_speckle_noise and end_folder == 'ultrasound':
                        dp.do_speckle_noise(x=x, y=y, file_prefix=file_prefix,
                                            filename=self.extract_name_from_path(filename, without_ch=False),
                                            end_folder=end_folder, path_to_augment=self.dir_augmented,
                                            path_to_params=self.dir_params, data_type=self.data_type,
                                            attention_mask=self.attention_mask,
                                            attention_input_dist=self.attention_input_dist)

                flipped_to_be_aug = dp.ret_all_files_in_folder(folder_path=self.dir_processed + '/augmented/flip/' +
                                                               end_folder, full_names=True)
                for filename in flipped_to_be_aug:
                    print('augmenting file', self.extract_name_from_path(filename, without_ch=False))
                    x = self.load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_low')
                    y = self.load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_high')

                    if self.do_blur and end_folder == 'ultrasound':
                        dp.do_blur(x=x, y=y, file_prefix=file_prefix,
                                   filename=self.extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder, path_to_augment=self.dir_augmented,
                                   path_to_params=self.dir_params, data_type=self.data_type,
                                   attention_mask=self.attention_mask, attention_input_dist=self.attention_input_dist)
                    if self.do_speckle_noise and end_folder == 'ultrasound':
                        dp.do_speckle_noise(x=x, y=y, file_prefix=file_prefix,
                                            filename=self.extract_name_from_path(filename, without_ch=False),
                                            end_folder=end_folder, path_to_augment=self.dir_augmented,
                                            path_to_params=self.dir_params, data_type=self.data_type,
                                            attention_mask=self.attention_mask,
                                            attention_input_dist=self.attention_input_dist)
        elif self.data_type == self.accepted_data_types[2]:
            if self.data_type == 'bi' and self.do_deform:
                print('No deform augmentation for bi SoS data')
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

                for filename in to_be_aug_files:
                    print('augmenting file', self.extract_name_from_path(filename, without_ch=False))
                    x = self.load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_low')
                    y = self.load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_high')
                    if self.do_flip:
                        dp.do_flip(x=x, y=y, file_prefix=file_prefix,
                                   filename=self.extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder,
                                   path_to_augment=self.dir_augmented)
                    if self.do_blur and end_folder == 'ultrasound':
                        dp.do_blur(x=x, y=y, file_prefix=file_prefix,
                                   filename=self.extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder, path_to_augment=self.dir_augmented,
                                   path_to_params=self.dir_params, data_type=self.data_type,
                                   attention_mask=self.attention_mask, attention_input_dist=self.attention_input_dist)
                    if self.do_speckle_noise and end_folder == 'ultrasound':
                        dp.do_speckle_noise(x=x, y=y, file_prefix=file_prefix,
                                            filename=self.extract_name_from_path(filename, without_ch=False),
                                            end_folder=end_folder, path_to_augment=self.dir_augmented,
                                            path_to_params=self.dir_params, data_type=self.data_type,
                                            attention_mask=self.attention_mask,
                                            attention_input_dist=self.attention_input_dist)

                flipped_to_be_aug = dp.ret_all_files_in_folder(folder_path=self.dir_processed + '/augmented/flip/' +
                                                               end_folder, full_names=True)
                for filename in flipped_to_be_aug:
                    print('augmenting file', self.extract_name_from_path(filename, without_ch=False))
                    x = self.load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_low')
                    y = self.load_file_to_numpy(full_file_name=filename, image_sign=file_prefix + '_high')

                    if self.do_blur and end_folder == 'ultrasound':
                        dp.do_blur(x=x, y=y, file_prefix=file_prefix,
                                   filename=self.extract_name_from_path(filename, without_ch=False),
                                   end_folder=end_folder, path_to_augment=self.dir_augmented,
                                   path_to_params=self.dir_params, data_type=self.data_type,
                                   attention_mask=self.attention_mask, attention_input_dist=self.attention_input_dist)
                    if self.do_speckle_noise and end_folder == 'ultrasound':
                        dp.do_speckle_noise(x=x, y=y, file_prefix=file_prefix,
                                            filename=self.extract_name_from_path(filename, without_ch=False),
                                            end_folder=end_folder, path_to_augment=self.dir_augmented,
                                            path_to_params=self.dir_params, data_type=self.data_type,
                                            attention_mask=self.attention_mask,
                                            attention_input_dist=self.attention_input_dist)

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
                self._clear_dir(path_to_dir=path_to_dir)

    def _clear_dir(self, path_to_dir):
        file_list = os.listdir(path_to_dir)
        for f in file_list:
            os.remove(os.path.join(path_to_dir, f))

    ##################################################################
    # ###### fit PCA for OA data      ####################################
    ##################################################################

    def fit_pca_on_targets(self, train_set, fit_ratio, num_components, pca_batch_size = 100):
        # fits pca on target images in the train_set. subsamples images with fit_ratio


        indices_of_sample = random.sample(range(len(train_set)), int(len(train_set)*fit_ratio))
        sample_files = [train_set[i] for i in indices_of_sample]
        targets = np.array(np.array([np.array(self.load_file_to_numpy(full_file_name=file_name,
            image_sign='OA' + '_high')) for file_name in sample_files]))

        # truncate targets
        targets = self.truncate_images_in_batch(batch=targets, trunc_points=self.trunc_points_before_pca)

        if self.oa_do_scale_center_before_pca:
            # get normalization parameters
            var_low, var_high = self.load_params(param_type="scale_params_before_pca")
            mean_low, mean_high = self.load_params(param_type="mean_images_before_pca")
            # scale targets
            targets = self.scale_and_center(targets, scale_params=var_high, mean_image=mean_high)
        n_channels = targets.shape[-1]
        targets = targets.reshape(-1,n_channels)
        print('fits pca')
        pca_model = IncrementalPCA(n_components = num_components, batch_size=pca_batch_size)
        pca_model.fit(targets)
        with open(self.dir_params + '/PCA' + '/OA_pca_model.sav', 'wb') as handle:
            pickle.dump(pca_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##################################################################
    # ###### do PCA and save data      ###############################
    ##################################################################

    def do_pca_and_save_data(self, file_paths):
        print('transforms data into pca coefficients and save the new data')
        # load model
        model = self.load_pca_model()
        if self.oa_do_scale_center_before_pca:
            # load normalization parameters
            var_low, var_high = self.load_params(param_type="scale_params_before_pca")
            mean_low, mean_high = self.load_params(param_type="mean_images_before_pca")
        for file_name in file_paths:
            image_high = self.load_file_to_numpy(full_file_name=file_name,image_sign='OA' + '_high')
            if self.oa_do_scale_center_before_pca:
                image_high = self.scale_and_center(image_high, scale_params=var_high, mean_image=mean_high)

            image_low = self.load_file_to_numpy(full_file_name=file_name, image_sign='OA' + '_low')
            if self.oa_do_scale_center_before_pca:
                image_low = self.scale_and_center(image_low, scale_params=var_low, mean_image=mean_low)

            if self.pca_use_regress:
                transformed_low = lu(image_low, spectra=model.components_)
                transformed_high = lu(image_high, spectra=model.components_)
            else:
                im_shape = image_high.shape
                image_high = image_high.reshape(-1, im_shape[-1])
                transformed_high = model.transform(image_high)
                new_shape = list(im_shape[:2])
                new_shape.append(model.n_components)
                transformed_high = transformed_high.reshape(new_shape)
                im_shape = image_low.shape
                image_low = image_low.reshape(-1, im_shape[-1])
                transformed_low = model.transform(image_low)
                new_shape = list(im_shape[:2])
                new_shape.append(model.n_components)
                transformed_low = transformed_low.reshape(new_shape)
            short_file_name = file_name.rsplit('/', 1)[-1] # get only file name without rest of path
            dic = {'OA_low_' + short_file_name: transformed_low, 'OA_high_' + short_file_name: transformed_high}
            pickle.dump(dic, open(file_name, 'wb'))

    def load_pca_model(self, path=None):
        if path is None:
            model = pickle.load(open(self.dir_params + '/PCA' + '/OA_pca_model.sav', 'rb'))
        else:
            model = pickle.load(open(path + '/OA_pca_model.sav', 'rb'))
        return model

    ##################################################################
    # ###### Normalization parameters ################################
    ##################################################################

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

        return n_obs, mean, var

    def _get_scale_center(self, before_pca=False):
        print('Calculates scaling parameters')

        if self.oa_do_scale_center_before_pca and before_pca:
            trunc_points = self.trunc_points_before_pca
        else:
            trunc_points = self.trunc_points

        # Initialize values

        if self.data_type == 'homo':
            mean_low = 0
            var_low = 0
            count_low = 0
            mean_high = 0
            var_high = 0
            count_high = 0
            for file in self.train_file_names:
                image_sign = self.image_type + '_high'
                image_high = self.load_file_to_numpy(file, image_sign)
                image_sign = self.image_type + '_low'
                image_low = self.load_file_to_numpy(file, image_sign)
                # truncating images
                if self.image_type == 'OA':
                    lq = np.quantile(image_low, trunc_points)
                    hq = np.quantile(image_high, trunc_points)
                    image_low = image_low.clip(lq[0], lq[1])
                    image_high = image_high.clip(hq[0], hq[1])
                # update values
                low_aggr = self.update_mean_var((count_low, mean_low, var_low), image_low)
                (count_low, mean_low, var_low) = low_aggr
                high_aggr = self.update_mean_var((count_high, mean_high, var_high), image_high)
                (count_high, mean_high, var_high) = high_aggr
        elif self.data_type == 'hetero' or self.data_type == 'bi':
            if self.hetero_mask_to_mask:
                mean_low = 0
                var_low = 0
                count_low = 0
                mean_high = 0
                var_high = 0
                count_high = 0
                for file in self.train_file_names:
                    image_sign = self.image_type + '_high'
                    image_high = self.load_file_to_numpy(file, image_sign)
                    image_sign = self.image_type + '_low'
                    image_low = self.load_file_to_numpy(file, image_sign)
                    # update values
                    low_aggr = self.update_mean_var((count_low, mean_low, var_low), image_low)
                    (count_low, mean_low, var_low) = low_aggr
                    high_aggr = self.update_mean_var((count_high, mean_high, var_high), image_high)
                    (count_high, mean_high, var_high) = high_aggr
            else:
                if self.attention_mask == 'simple':
                    mean_low = [0, 0, 0]
                    var_low = [0, 0, 0]
                    count_low = 0
                    mean_high = 0
                    var_high = 0
                    count_high = 0
                    for file in self.train_file_names:
                        image_sign = self.image_type + '_high'
                        image_high = self.load_file_to_numpy(file, image_sign)
                        image_sign = self.image_type + '_low'
                        image_low = self.load_file_to_numpy(file, image_sign)
                        image_low_image1 = image_low[:, :, 0]
                        image_low_image2 = image_low[:, :, self.attention_input_dist[0]]  # take the first tissue sos
                        image_low_sos = image_low[:, :, np.sum(self.attention_input_dist):]
                        # update values
                        low_aggr_image1 = self.update_mean_var((count_low, mean_low[0], var_low[0]), image_low_image1)
                        low_aggr_image2 = self.update_mean_var((count_low, mean_low[1], var_low[1]), image_low_image2)
                        (count_low, mean_low[0], var_low[0]) = low_aggr_image1
                        (count_low, mean_low[1], var_low[1]) = low_aggr_image2
                        low_aggr_sos = self.update_mean_var((count_low, mean_low[1], var_low[1]), image_low_sos)
                        (count_low, mean_low[2], var_low[2]) = low_aggr_sos
                        high_aggr = self.update_mean_var((count_high, mean_high, var_high), image_high)
                        (count_high, mean_high, var_high) = high_aggr
                    '''if self.attention_mask == 'simple' or self.attention_mask == 'complex':
                        mean_low = [0, 0, 0]
                        var_low = [0, 0, 0]
                        count_low = 0
                        mean_high = 0
                        var_high = 0
                        count_high = 0
                        for file in self.train_file_names:
                            image_sign = self.image_type + '_high'
                            image_high = self.load_file_to_numpy(file, image_sign)
                            image_sign = self.image_type + '_low'
                            image_low = self.load_file_to_numpy(file, image_sign)
                            image_low_image1 = image_low[:, :, 0]
                            image_low_image2 = image_low[:, :, 1]
                            image_low_sos = image_low[:, :, 2:]
                            # update values
                            low_aggr_image1 = self.update_mean_var((count_low, mean_low[0], var_low[0]), image_low_image1)
                            low_aggr_image2 = self.update_mean_var((count_low, mean_low[1], var_low[1]), image_low_image2)
                            (count_low, mean_low[0], var_low[0]) = low_aggr_image1
                            (count_low, mean_low[1], var_low[1]) = low_aggr_image2
                            low_aggr_sos = self.update_mean_var((count_low, mean_low[1], var_low[1]), image_low_sos)
                            (count_low, mean_low[2], var_low[2]) = low_aggr_sos
                            high_aggr = self.update_mean_var((count_high, mean_high, var_high), image_high)
                            (count_high, mean_high, var_high) = high_aggr'''
                else:
                    mean_low = [0,0]
                    var_low = [0,0]
                    count_low = 0
                    mean_high = 0
                    var_high = 0
                    count_high = 0
                    for file in self.train_file_names:
                        image_sign = self.image_type + '_high'
                        image_high = self.load_file_to_numpy(file, image_sign)
                        image_sign = self.image_type + '_low'
                        image_low = self.load_file_to_numpy(file, image_sign)
                        image_low_image = image_low[:,:,0]
                        image_low_sos = image_low[:,:,1:]
                        # update values
                        low_aggr_image = self.update_mean_var((count_low, mean_low[0], var_low[0]), image_low_image)
                        (count_low, mean_low_image, var_low_image) = low_aggr_image
                        mean_low[0] = mean_low_image
                        var_low[0] = var_low_image
                        low_aggr_sos = self.update_mean_var((count_low, mean_low[1], var_low[1]), image_low_sos)
                        (count_low, mean_low_sos, var_low_sos) = low_aggr_sos
                        mean_low[1] = mean_low_sos
                        var_low[1] = var_low_sos
                        high_aggr = self.update_mean_var((count_high, mean_high, var_high), image_high)
                        (count_high, mean_high, var_high) = high_aggr
        else:
            # now we are in the bi case; this should be redundant later on!
            if self.attention_mask == 'simple':
                mean_low = [0, 0, 0]
                var_low = [0, 0, 0]
                count_low = 0
                mean_high = 0
                var_high = 0
                count_high = 0
                for file in self.train_file_names:
                    image_sign = self.image_type + '_high'
                    image_high = self.load_file_to_numpy(file, image_sign)
                    image_sign = self.image_type + '_low'
                    image_low = self.load_file_to_numpy(file, image_sign)
                    image_low_image1 = image_low[:, :, 0]
                    image_low_image2 = image_low[:, :, self.attention_input_dist[0]]  # take the first tissue sos
                    image_low_sos = image_low[:, :, np.sum(self.attention_input_dist):]
                    # update values
                    low_aggr_image1 = self.update_mean_var((count_low, mean_low[0], var_low[0]), image_low_image1)
                    low_aggr_image2 = self.update_mean_var((count_low, mean_low[1], var_low[1]), image_low_image2)
                    (count_low, mean_low[0], var_low[0]) = low_aggr_image1
                    (count_low, mean_low[1], var_low[1]) = low_aggr_image2
                    low_aggr_sos = self.update_mean_var((count_low, mean_low[1], var_low[1]), image_low_sos)
                    (count_low, mean_low[2], var_low[2]) = low_aggr_sos
                    high_aggr = self.update_mean_var((count_high, mean_high, var_high), image_high)
                    (count_high, mean_high, var_high) = high_aggr
        scale_params_low = var_low
        scale_params_high = var_high
        print(scale_params_low, scale_params_high)
        print(mean_low, mean_high)

        if before_pca:
            file_suffix = '_before_pca'
        else:
            file_suffix = ''
        # construct dictionaries and save parameters
        if self.image_type == 'US':
            us_scale_params = {'US_low': scale_params_low, 'US_high': scale_params_high}
            us_mean_images = {'US_low': mean_low, 'US_high': mean_high,}
            with open(self.dir_params + '/scale_and_center' + '/US_scale_params', 'wb') as handle:
                pickle.dump(us_scale_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.dir_params + '/scale_and_center' + '/US_mean_images', 'wb') as handle:
                pickle.dump(us_mean_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            oa_scale_params = {'OA_low': scale_params_low, 'OA_high': scale_params_high, 'trunc_points': trunc_points}
            oa_mean_images = {'OA_low': mean_low, 'OA_high': mean_high, 'trunc_points': trunc_points}
            with open(self.dir_params + '/scale_and_center' + '/OA_scale_params' + file_suffix, 'wb') as handle:
                pickle.dump(oa_scale_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.dir_params + '/scale_and_center' + '/OA_mean_images' + file_suffix, 'wb') as handle:
                pickle.dump(oa_mean_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

        ##################################################################
        # ###### Data Normalization ######################################
        ##################################################################

    def scale_and_center(self, batch, scale_params, mean_image):
        """ scales and centers the input batch given the scale_params and the mean_image
                input: batch: for US (N,H,W) array, for OA: (N,H,W,C) array
                       scale_params: minimum and maximum over the data set (channel by channel)
                            for US: array of shape (2,), for OA: arrays of shape (2,C)
                        mean_image: mean image for the whole data set
                            for US: array of shape (H,W), for OA: (H,W,C)
                       image_type: 'US' or 'OA'
                output: batch_out array with same shape as batch"""
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
        batch_out = batch*np.sqrt(scale_params) + mean_image
        return batch_out

    def load_params(self, param_type, dir_params=None, trunc_points=None):
        """ loads the specified parameters from file
            input: image_type: 'US' or 'OA'
                    param_type: 'scale' or 'mean_image' (maybe more options later)
            output: params_low, params_high: the parameters."""
        # dir_params = '/mnt/local/mounted'
        if not self.do_scale_center and param_type in ['scale_params', 'mean_images']:
            print('No Scaling done due to parameter do_scale_center=False.')
            return None, None
        if not self.oa_do_scale_center_before_pca and param_type in ['scale_params_before_pca', 'mean_images_before_pca']:
            print('No Scaling done due to parameter oa_do_scale_center_before_pca=False.')
            return None, None
        if dir_params is None:
            dir_params = self.dir_params + '/' + 'scale_and_center/'
        if param_type in ['scale_params', 'mean_images', 'scale_params_before_pca', 'mean_images_before_pca']:
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
            if trunc_points is None:
                params_trunc_points = params['trunc_points']
            else:
                params_trunc_points = trunc_points
            if param_type in ['scale_params_before_pca', 'mean_images_before_pca']:
                trunc_points_to_compare = self.trunc_points_before_pca
            else:
                trunc_points_to_compare = self.trunc_points
            if trunc_points_to_compare != params_trunc_points:
                sys.exit('Truncation of saved parameters does not fit the chosen truncation. '
                         'Truncation of saved parameters is ' + str(params_trunc_points))
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
            if self.data_type=='homo':
                scale_center_x_val = self.scale_and_center(x, scale_params_low, mean_image_low)
                scale_center_y_val = self.scale_and_center(y, scale_params_high, mean_image_high)

                '''elif self.only_f_test_in_target:
                    x_ftest = x[:, :, :, 0]
                    scale_center_x_ftest = self.scale_and_center(x_ftest, scale_params_low[0], mean_image_low[0])
                    if self.include_regression_error:
                        x_image = x[:, :, :, 1:5]
                        x_error = x[:, :, :, 5]
                        scale_center_x_image = self.scale_and_center(x_image, scale_params_low[1], mean_image_low[1])
                        scale_center_x_error = self.scale_and_center(x_error, scale_params_low[2], mean_image_low[2])
                        scale_center_x_val = np.empty(x.shape)
                        scale_center_x_val[:, :, :, 0] = scale_center_x_ftest
                        scale_center_x_val[:, :, :, 1:5] = scale_center_x_image
                        scale_center_x_val[:, :, :, 5] = scale_center_x_error
                    else:
                        sys.exit('You should include the regression error, or run it without scale and center,'
                                 ' or implement this case as well')
    
                    scale_center_y_val = self.scale_and_center(y, scale_params_high, mean_image_high)'''
            elif self.data_type=='hetero' or self.data_type=='bi':
                if self.hetero_mask_to_mask:
                    scale_center_x_val = self.scale_and_center(x, scale_params_low, mean_image_low)
                    scale_center_y_val = self.scale_and_center(y, scale_params_high, mean_image_high)
                elif self.attention_mask == 'simple' or self.attention_mask == 'complex':
                    x_image1 = x[:, :, :, 0:self.attention_input_dist[0]]
                    x_image2 = x[:, :, :, self.attention_input_dist[0]:np.sum(self.attention_input_dist)]
                    x_sos = x[:, :, :, np.sum(self.attention_input_dist):]

                    scale_center_x_image1 = self.scale_and_center(x_image1, scale_params_low[0], mean_image_low[0])
                    scale_center_x_image2 = self.scale_and_center(x_image2, scale_params_low[1], mean_image_low[1])
                    scale_center_x_sos = self.scale_and_center(x_sos, scale_params_low[2], mean_image_low[2])

                    scale_center_x_val = np.empty(x.shape)
                    scale_center_x_val[:, :, :, 0:self.attention_input_dist[0]] = scale_center_x_image1
                    scale_center_x_val[:, :, :,
                    self.attention_input_dist[0]:np.sum(self.attention_input_dist)] = scale_center_x_image2
                    scale_center_x_val[:, :, :, np.sum(self.attention_input_dist):] = scale_center_x_sos

                    scale_center_y_val = self.scale_and_center(y, scale_params_high, mean_image_high)
                    '''x_image1 = x[:, :, :, 0]
                    x_image2 = x[:, :, :, 1]
                    x_sos = x[:, :, :, 2:]

                    scale_center_x_image1 = self.scale_and_center(x_image1, scale_params_low[0], mean_image_low[0])
                    scale_center_x_image2 = self.scale_and_center(x_image2, scale_params_low[1], mean_image_low[1])
                    scale_center_x_sos = self.scale_and_center(x_sos, scale_params_low[2], mean_image_low[2])

                    scale_center_x_val = np.empty(x.shape)
                    scale_center_x_val[:, :, :, 0] = scale_center_x_image1
                    scale_center_x_val[:, :, :, 1] = scale_center_x_image2
                    scale_center_x_val[:, :, :, 2:] = scale_center_x_sos

                    scale_center_y_val = self.scale_and_center(y, scale_params_high, mean_image_high)'''
                else:
                    x_image = x[:, :, :, 0]
                    x_sos = x[:,:,:,1:]
                    scale_center_x_image = self.scale_and_center(x_image, scale_params_low[0], mean_image_low[0])
                    scale_center_x_sos = self.scale_and_center(x_sos, scale_params_low[1], mean_image_low[1])
                    scale_center_x_val = np.empty(x.shape)
                    scale_center_x_val[:, :, :, 0] = scale_center_x_image
                    scale_center_x_val[:, :, :, 1:] = scale_center_x_sos
                    scale_center_y_val = self.scale_and_center(y, scale_params_high, mean_image_high)
            else:
                if self.attention_mask == 'simple' or self.attention_mask == 'complex':
                    x_image1 = x[:, :, :, 0:self.attention_input_dist[0]]
                    x_image2 = x[:, :, :, self.attention_input_dist[0]:np.sum(self.attention_input_dist)]
                    x_sos = x[:, :, :, np.sum(self.attention_input_dist):]

                    scale_center_x_image1 = self.scale_and_center(x_image1, scale_params_low[0], mean_image_low[0])
                    scale_center_x_image2 = self.scale_and_center(x_image2, scale_params_low[1], mean_image_low[1])
                    scale_center_x_sos = self.scale_and_center(x_sos, scale_params_low[2], mean_image_low[2])

                    scale_center_x_val = np.empty(x.shape)
                    scale_center_x_val[:, :, :, 0:self.attention_input_dist[0]] = scale_center_x_image1
                    scale_center_x_val[:, :, :, self.attention_input_dist[0]:np.sum(self.attention_input_dist)] = scale_center_x_image2
                    scale_center_x_val[:, :, :, np.sum(self.attention_input_dist):] = scale_center_x_sos

                    scale_center_y_val = self.scale_and_center(y, scale_params_high, mean_image_high)
                else:
                    sys.exit('No Scale and center for new kind of attention_mask implemented, check input or code.')

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
