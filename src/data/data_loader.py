import sys
import numpy as np
from pathlib import Path
import os
import scipy.io
import pickle
import random
import data.augmentation


class ProcessData(object):
    """
    TODO: Description of data loader in general; keep it growing
    train_ratio:    Gives the train and validation split for the model
    process_raw:    Process the raw data in the input folder and load them into processed folder
    image_type:     Either 'US' or 'OA' to select which data should be loaded
    do_augment:     Do the augmentation and save them in the correspodning folders
    """

    def __init__(self,
                 train_ratio,
                 image_type,
                 do_augment=False,
                 process_raw_data=False,
                 do_flip=False,
                 do_deform=False,
                 do_blur=False,
                 get_scale_center = False):
        # initialize and write into self, then call the prepare data and return the data to the trainer
        self.train_ratio = train_ratio
        self.do_augment = do_augment
        self.do_flip = do_flip
        self.do_deform = do_deform
        self.do_blur = do_blur

        self.image_type = image_type

        project_root_dir = Path().resolve().parents[1]  # root directory
        self.dir_raw_in = project_root_dir / 'data' / 'raw' / 'new_in'
        self.dir_processed_all = project_root_dir / 'data' / 'processed' / 'processed_all'
        self.dir_processed = project_root_dir / 'data' / 'processed'
        self.all_folder = False  # check if raw folder was already processed
        self.process_oa = True  # process raw oa data
        self.process_us = True  # process raw us data
        self.process_raw = process_raw_data  # call method _process_raw_data
        self.get_scale_center = get_scale_center # get scaling and mean image and store them
        self.dir_params = project_root_dir/ 'params'


        # run _prepare_data which calls all other needed methods
        self._prepare_data()

        # self.X_train, self.y_train, self.X_test, self.y_test, self.df_complete = self._prepare_data()

    def _prepare_data(self):
        # process data
        if self.process_raw:
            self._process_raw_data()

        if self.do_augment:
            self.augment_data()

        # get the original file names, split them up to validation and training and write them into self
        self.original_file_names = self._retrieve_original_file_names()
        self.train_file_names, self.val_file_names = self._train_val_split(original_file_names=self.original_file_names)
        self._add_augmented_file_names_to_train()
        if self.get_scale_center:
            self._get_scale_center()

        self.X_val, self.Y_val = self._load_processed_data(full_file_names=self.val_file_names)

    def batch_names(self, batch_size):
        # give a list and return the corresponding batch names
        self.train_batch_chunks = np.array_split(np.array(self.train_file_names),
                                                 int(len(self.train_file_names) / batch_size))
        self.batch_number = len(self.train_batch_chunks)

    def create_train_batches(self, batch_names):
        # return the batches
        X, Y = self._load_processed_data(full_file_names=batch_names)
        return X, Y

    def _process_raw_data(self):
        # load the raw data in .mat format, split up the us and oa and load them in dictionaries into processed folder
        in_directories = [s for s in os.listdir(self.dir_raw_in) if '.' not in s]
        print("Preprocess raw data")
        if not self.all_folder:
            for sub in in_directories:
                if (next((True for s in os.listdir(self.dir_processed_all / 'ultrasound') if sub in s), False)
                    and next((True for s in os.listdir(self.dir_processed_all / 'optoacoustic') if sub in s), False)):
                    in_directories.remove(sub)

        for chunk_folder in in_directories:
            sample_directories = [s for s in os.listdir(self.dir_raw_in / chunk_folder) if '.' not in s]
            print("Processing data from raw input folder: " + chunk_folder)

            for sample_folder in sample_directories:
                in_files = os.listdir(self.dir_raw_in / chunk_folder / sample_folder)
                us_file = [s for s in in_files if 'US_' in s]
                oa_file = [s for s in in_files if 'OA_' in s]
                if us_file and self.process_us:
                    us_raw = scipy.io.loadmat(self.dir_raw_in / chunk_folder / sample_folder / us_file[0])
                    for i in range(us_raw['US_low'].shape[2]):
                        name_us_low = 'US_low_' + chunk_folder + '_' + sample_folder + '_ch' + str(i)
                        name_us_high = 'US_high_' + chunk_folder + '_' + sample_folder + '_ch' + str(i)  #
                        name_us_save = 'US_' + chunk_folder + '_' + sample_folder + '_ch' + str(i)
                        dict_us_single = {name_us_low: us_raw['US_low'][:, :, i],
                                          name_us_high: us_raw['US_high'][:, :, i]}
                        self._save_dict_with_pickle(file=dict_us_single,
                                                    folder_name='ultrasound', file_name=name_us_save)

                if oa_file and self.process_oa:
                    oa_raw = scipy.io.loadmat(self.dir_raw_in / chunk_folder / sample_folder / oa_file[0])
                    name_oa_low = 'OA_low_' + chunk_folder + '_' + sample_folder
                    name_oa_high = 'OA_high_' + chunk_folder + '_' + sample_folder
                    name_oa_save = 'OA_' + chunk_folder + '_' + sample_folder
                    dict_oa = {name_oa_low: oa_raw['OA_low'],
                               name_oa_high: oa_raw['OA_high']}
                    self._save_dict_with_pickle(file=dict_oa, folder_name='optoacoustic', file_name=name_oa_save)

    def _train_val_split(self, original_file_names):
        # TODO: unify this seed with all other seeds
        random.seed(42)
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
        file_names = self._names_to_list(folder_name=self.dir_processed_all / end_folder, name_list=file_names)
        return file_names

    def _add_augmented_file_names_to_train(self):
        # add the file names of the augmented data to self.train_file_names
        path_augmented = self.dir_processed / 'augmented'
        if self.do_augment:
            if self.do_blur:
                self.train_file_names = self._names_to_list(folder_name=path_augmented / 'blur',
                                                            name_list=self.train_file_names)
            if self.do_deform:
                self.train_file_names = self._names_to_list(folder_name=path_augmented / 'deform',
                                                            name_list=self.train_file_names)
            if self.do_flip:
                self.train_file_names = self._names_to_list(folder_name=path_augmented / 'flip',
                                                            name_list=self.train_file_names)



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
        with open(self.dir_processed_all / folder_name / file_name, 'wb') as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def _get_scale_center(self):
        # Initialize values
        i = 0
        if self.image_type == 'US':
            temp = 1
            shape_for_mean_image = [401,401]
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
            with open(self.dir_params / 'scale_and_center' / 'US_scale_params', 'wb') as handle:
                pickle.dump(US_scale_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.dir_params / 'scale_and_center' / 'US_mean_images', 'wb') as handle:
                pickle.dump(US_mean_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            OA_scale_params = {'OA_low': [min_data_low, max_data_low], 'OA_high': [min_data_high, max_data_high]}
            OA_mean_images = {'OA_low': mean_image_low, 'OA_high': mean_image_high}
            ### CAUTION: the OA mean image gets stored in the (C,N,N) shape!!!! (that's what the moveaxis is doing)
            with open(self.dir_params / 'scale_and_center' / 'OA_scale_params', 'wb') as handle:
                pickle.dump(OA_scale_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.dir_params / 'scale_and_center' / 'OA_mean_images', 'wb') as handle:
                pickle.dump(OA_mean_images, handle, protocol=pickle.HIGHEST_PROTOCOL)




    def augment_data(self, augment_oa = False, augment_us = False):
        print("Augment Data is not doing anything yet.")
        pass
