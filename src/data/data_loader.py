import sys
import numpy as np
from pathlib import Path
import os
import scipy.io
import pickle
import data.augmentation


class ProcessData(object):
    """
    TODO: Description of data loader in general; keep it growing
    train_ratio:    Gives the train and validation split for the model
    process_raw:    Process the raw data in the input folder and load them into processed folder
    image_type:     Either 'US' or 'OA' to select which data should be loaded
    """

    def __init__(self, train_ratio, image_type, process_raw_data=False, do_flip = False, do_deform = False, do_blur = False):
        # initialize and write into self, then call the prepare data and return the data to the trainer
        self.train_ratio = train_ratio
        self.do_flip = do_flip
        self.do_deform = do_deform
        self.do_blur = do_blur

        self.image_type = image_type

        project_root_dir = Path().resolve().parents[1]  # root directory
        self.dir_raw_in = project_root_dir / 'data' / 'raw' / 'new_in'
        print(project_root_dir)
        self.dir_processed = project_root_dir / 'data' / 'processed' / 'processed_all'
        self.all_folder = False  # check if raw folder was already processed
        self.process_oa = True  # process raw oa data
        self.process_us = True  # process raw us data
        self.process_raw = process_raw_data  # call method _process_raw_data

        # run _prepare_data which calls all other needed methods
        self._prepare_data()

        # self.X_train, self.y_train, self.X_test, self.y_test, self.df_complete = self._prepare_data()

    def _prepare_data(self):
        # process data
        if self.process_raw:
            self._process_raw_data()

        X,Y = self._load_processed_data()
        print(X.shape)
        print(Y.shape)
        # return X_train, y_train, X_test, y_test, df_complete

    def _process_raw_data(self):
        # load the raw data in .mat format, split up the us and oa and load them in dictionaries into processed folder
        in_directories = [s for s in os.listdir(self.dir_raw_in) if '.' not in s]
        print("Preprocess raw data")
        if not self.all_folder:
            for sub in in_directories:
                if (next((True for s in os.listdir(self.dir_processed / 'ultrasound') if sub in s), False)
                    and next((True for s in os.listdir(self.dir_processed / 'optoacoustic') if sub in s), False)):
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
                        with open(self.dir_processed / 'ultrasound' / name_us_save, 'wb') as handle:
                            pickle.dump(dict_us_single, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        #np.save(self.dir_processed / 'ultrasound' / name_us_save, dict_us_single)
                    #handle.close()

                if oa_file and self.process_oa:
                    oa_raw = scipy.io.loadmat(self.dir_raw_in / chunk_folder / sample_folder / oa_file[0])
                    name_oa_low = 'OA_low_' + chunk_folder + '_' + sample_folder
                    name_oa_high = 'OA_high_' + chunk_folder + '_' + sample_folder
                    name_oa_save = 'OA_' + chunk_folder + '_' + sample_folder
                    dict_oa = {name_oa_low: oa_raw['OA_low'],
                               name_oa_high: oa_raw['OA_high']}
                    with open(self.dir_processed / 'optoacoustic' / name_oa_save, 'wb') as f:
                        pickle.dump(dict_oa, f, protocol=pickle.HIGHEST_PROTOCOL)
                    #np.save(self.dir_processed / 'optoacoustic' / name_oa_save, dict_oa)
                #f.close()

    def _split_data(self, df):
        train_size = int(len(df) * self.train_ratio)
        df_train, df_test = df[:train_size], df[train_size:]
        return df_train, df_test

    def _partition_for_cnn(self, df):
        # returns the data in the format for the model
        pass
        # return ...

    def _load_processed_data(self):
        # load the already preprocessed data and store it into X (low) and Y (high)
        if self.image_type not in ['US','OA']:
            sys.exit("Error: No valid image_type selected!")
        else:
            if(self.image_type=='US'):
                end_folder = 'ultrasound'
            else:
                end_folder = 'optoacoustic'
            in_files = [s for s in os.listdir(self.dir_processed/end_folder) if '.DS_' not in s]
            print(self.dir_processed / end_folder)
            X = np.array(
                [np.array(self._load_file_to_numpy(folder_name=self.dir_processed / end_folder,
                                                   file_name=fname,
                                                   image_sign=self.image_type + '_low')) for fname in in_files])
            Y = np.array(
                [np.array(self._load_file_to_numpy(folder_name=self.dir_processed/end_folder,
                                                   file_name=fname,
                                                   image_sign=self.image_type + '_high')) for fname in in_files])

        return X, Y

    def _load_file_to_numpy(self,folder_name, file_name, image_sign):
        # helper function to load and read the data
        with open(folder_name / file_name, 'rb') as handle:
            sample = pickle.load(handle)

        sample_array = [value for key, value in sample.items() if image_sign in key][0]

        return sample_array

    def augment_data(self, augment_oa = False, augment_us = False):
        pass

