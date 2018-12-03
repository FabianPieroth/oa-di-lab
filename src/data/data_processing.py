import pickle
import data.augmentation
import scipy.io
import numpy as np


# this File contains several helper functions

# Pre-processing


# homo

def pre_us_homo(new_in_folder, study_folder, filename, scan_num, save_folder):
    us_raw = scipy.io.loadmat(new_in_folder + '/' + study_folder + '/' +
                              scan_num + '/' + filename)

    # take random channel as training set
    j = np.random.randint(0, us_raw['US_low'].shape[2])
    name_us_low = 'US_low_' + study_folder + '_' + scan_num + '_ch' + str(j)
    name_us_high = 'US_high_' + study_folder + '_' + scan_num + '_ch' + str(j)  #
    name_us_save = 'US_' + study_folder + '_' + scan_num + '_ch' + str(j)
    dict_us_single = {name_us_low: us_raw['US_low'][:, :, j],
                      name_us_high: us_raw['US_high'][:, :, j]}
    save_dict_with_pickle(file=dict_us_single,
                          folder_name=save_folder + '/ultrasound', file_name=name_us_save)


def pre_oa_homo(new_in_folder, study_folder, filename, scan_num, save_folder, cut_half=True, height_channel=200):
    oa_raw = scipy.io.loadmat(new_in_folder + '/' + study_folder + '/' +
                              scan_num + '/' + filename)
    name_oa_low = 'OA_low_' + study_folder + '_' + scan_num
    name_oa_high = 'OA_high_' + study_folder + '_' + scan_num
    name_oa_save = 'OA_' + study_folder + '_' + scan_num + '_ch0'
    if cut_half:
        oa_low = oa_raw['OA_low'][:height_channel,:,:]
        oa_high = oa_raw['OA_high'][:height_channel,:,:]
    else:
        oa_low = oa_raw['OA_low']
        oa_high = oa_raw['OA_high']
    dict_oa = {name_oa_low: oa_low,
               name_oa_high: oa_high}
    save_dict_with_pickle(file=dict_oa, folder_name=save_folder + '/optoacoustic',
                          file_name=name_oa_save)

# File Saving and Loading


def save_dict_with_pickle(file, folder_name, file_name):
    # use this to save pairs of low and high quality pictures
    with open(folder_name + '/' + file_name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Augmentations


def do_flip(x, y, file_prefix, filename, end_folder, path_to_augment):

    aug_x, aug_y = data.augmentation.flip(x, y)
    dict_save, name_save = create_file_names_and_dict(aug_x, aug_y, file_prefix, filename, aug_type='flip')
    save_dict_with_pickle(dict_save, path_to_augment + "/flip/" + end_folder, name_save)


def do_deform(x, y, file_prefix, filename, end_folder, path_to_augment, path_to_params, num_deform):

    aug_x, aug_y, params = data.augmentation.elastic_deform(x, y)
    dict_save, name_save = create_file_names_and_dict(aug_x, aug_y, file_prefix, filename, aug_type='deform')

    save_dict_with_pickle(dict_save, path_to_augment + "/deform/" + end_folder, name_save + str(num_deform))
    name_save_params = name_save + '_params'
    save_dict_with_pickle(params, path_to_params + '/augmentation/deform/' + end_folder, name_save_params)


def do_blur(x, y, file_prefix, filename, end_folder, path_to_augment, path_to_params):
    aug_x, aug_y, params = data.augmentation.blur(x, y,
                                                  lower_lim=1, upper_lim=3)

    dict_save, name_save = create_file_names_and_dict(aug_x, aug_y, file_prefix, filename, aug_type='blur')

    save_dict_with_pickle(dict_save, path_to_augment + "/blur/" + end_folder, name_save)
    name_save_params = name_save + '_params'
    save_dict_with_pickle(params, path_to_params + '/augmentation/blur/' + end_folder, name_save_params)


def do_crop(x, y, file_prefix, filename, end_folder, path_to_augment, path_to_params):
    aug_x, aug_y, params = data.augmentation.crop_stretch(x, y)

    dict_save, name_save = create_file_names_and_dict(aug_x, aug_y, file_prefix, filename, aug_type='crop')

    save_dict_with_pickle(dict_save, path_to_augment + "/crop/" + end_folder, name_save)
    name_save_params = name_save + '_params'
    save_dict_with_pickle(params, path_to_params + '/augmentation/crop/' + end_folder, name_save_params)


def do_rchannels(end_folder, filename, read_in_folder, num_channels, path_to_augment):
    dict_list, save_names = data.augmentation.rchannels(filename=filename,
                                                        dir_raw_in=read_in_folder,
                                                        num_rchannels=num_channels)
    for i in range(len(dict_list)):
        save_dict_with_pickle(dict_list[i], path_to_augment + "/rchannels/" + end_folder, save_names[i])


def create_file_names_and_dict(aug_x, aug_y, file_prefix, filename, aug_type):
    name_low = file_prefix + '_low_' + aug_type
    name_high = file_prefix + '_high_' + aug_type
    name_save = file_prefix + '_' + filename + '_' + aug_type

    dict_save = {name_low: aug_x,
                 name_high: aug_y}
    return dict_save, name_save
