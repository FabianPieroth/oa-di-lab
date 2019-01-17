import pickle
import data.augmentation
import scipy.io
import numpy as np
import os
from logger.oa_spectra_analysis.oa_for_DILab import linear_unmixing as lu
from logger.oa_spectra_analysis.oa_for_DILab import spectral_F_test as f_test

# this File contains several helper functions

# Pre-processing


def ret_all_files_in_folder(folder_path, full_names=True):
    files = [s for s in os.listdir(folder_path) if filter_hidden_files(s)]
    if full_names:
        files = [folder_path + '/' + s for s in files]
    return files


def filter_hidden_files(string):
    bool1 = ('.DS_' in string)
    bool2 = ('._' in string)

    return not any([bool1, bool2])

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


def pre_oa_homo(new_in_folder, study_folder, filename, scan_num, save_folder, cut_half=True, height_channel=201,
                regression_coefficients=None, use_regressed_oa=True, include_regression_error=False, add_f_test=False,
                only_f_test_in_target=False, channel_slice_oa=None):
    oa_raw = scipy.io.loadmat(new_in_folder + '/' + study_folder + '/' +
                              scan_num + '/' + filename)
    name_oa_low = 'OA_low_' + study_folder + '_' + scan_num
    name_oa_high = 'OA_high_' + study_folder + '_' + scan_num
    name_oa_save = 'OA_' + study_folder + '_' + scan_num + '_ch0'
    if cut_half:
        oa_low = oa_raw['OA_low'][:height_channel, :, :]
        oa_high = oa_raw['OA_high'][:height_channel, :, :]
    else:
        oa_low = oa_raw['OA_low']
        oa_high = oa_raw['OA_high']
    if use_regressed_oa:
        # print('Use the regression coefficients instead of the spectral information.')
        oa_low_regress = lu(oa_low, spectra=regression_coefficients, return_error=include_regression_error)
        oa_high_regress = lu(oa_high, spectra=regression_coefficients)
        if include_regression_error:
            # the error gets added to the input data
            oa_low_regress = np.concatenate((oa_low_regress[0], np.expand_dims(oa_low_regress[1], axis=2)), axis=2)
            # oa_high_regress = np.concatenate((oa_high_regress[0], np.expand_dims(oa_high_regress[1], axis=2)), axis=2)
        if add_f_test:
            # print('Additionally to the regressed data, we add the p_values of the f_test.')
            f_test_low = np.expand_dims(f_test(oa_low, spectra=regression_coefficients), axis=2)
            f_test_high = np.expand_dims(f_test(oa_high, spectra=regression_coefficients), axis=2)
            oa_low_regress = np.concatenate((f_test_low, oa_low_regress), axis=2)
            if only_f_test_in_target:
                oa_high_regress = f_test_high
                # print('We only take the p_values from the f_test as target image,
                # to only learn the significance mask.')
            else:
                oa_high_regress = np.concatenate((f_test_high, oa_high_regress), axis=2)
        oa_low = oa_low_regress
        oa_high = oa_high_regress
    else:
        # here you can choose which from the 28 channels you want to keep
        oa_low = oa_low[:, :, channel_slice_oa]
        oa_high = oa_high[:, :, channel_slice_oa]
        if add_f_test:
            # here you can add the f-test to the first channel together with the rest of the slice
            f_test_low = np.expand_dims(f_test(oa_low, spectra=regression_coefficients[:, channel_slice_oa]), axis=2)
            oa_low = np.concatenate((f_test_low, oa_low), axis=2)
        if only_f_test_in_target:
            # if this is true, only the f-test is in the target
            oa_high = np.expand_dims(f_test(oa_high, spectra=regression_coefficients[:, channel_slice_oa]), axis=2)

    dict_oa = {name_oa_low: oa_low,
               name_oa_high: oa_high}
    save_dict_with_pickle(file=dict_oa, folder_name=save_folder + '/optoacoustic',
                          file_name=name_oa_save)

# hetero


def pre_us_hetero(new_in_folder, study_folder, scan_num, filename_low, filename_high, save_folder, hetero_mask_to_mask,
                  attention_mask='Not'):

    us_raw_low = scipy.io.loadmat(new_in_folder + '/' + study_folder + '/' +
                                  scan_num + '/' + filename_low)
    us_raw_high = scipy.io.loadmat(new_in_folder + '/' + study_folder + '/' +
                                   scan_num + '/' + filename_high)

    single_sos = [np.float64(s) for s in us_raw_low['single_SoS'].flatten()]
    us_low_samples = us_raw_low['US_low_samples']

    couplant_sos = np.float64(us_raw_high['couplant_SoS'].flatten()[0])
    tissue_mask = np.array(us_raw_high['tissue_mask']).astype('float')
    tissue_sos = [np.float64(s) for s in us_raw_high['tissue_SoS'].flatten()]
    us_high_samples = us_raw_high['US_high_samples']
    # on which axis to expand the dimension of the numpy array
    common_axis = 2
    if not hetero_mask_to_mask and not attention_mask == 'complex':

        for low_channel in range(us_low_samples.shape[2]):

            # create channel with single sos parameter
            single_sos_channel = np.full(tissue_mask.shape, single_sos[low_channel])
            single_sos_channel = np.expand_dims(single_sos_channel, axis=common_axis)

            for high_channel in range(us_high_samples.shape[2]):

                # fill mask with sos parameters
                custom_mask = np.copy(tissue_mask)
                custom_mask[custom_mask == 0] = couplant_sos
                custom_mask[custom_mask == 1] = tissue_sos[high_channel]
                custom_mask = np.expand_dims(custom_mask, axis=common_axis)

                # create names and save
                name_us_low = 'US_low_' + study_folder + '_' + scan_num + '_ch' + str(low_channel)
                name_us_high = 'US_high_' + study_folder + '_' + scan_num + '_ch' + str(high_channel)
                name_us_save = 'US_' + study_folder + '_' + scan_num + '_ch' + str(low_channel) + 'and' + str(high_channel)

                us_low_ex_dim = np.expand_dims(us_low_samples[:,:,low_channel], axis=common_axis)

                if attention_mask == 'simple':
                    us_low_save = np.concatenate((us_low_ex_dim, us_low_ex_dim, custom_mask, single_sos_channel),
                                                 axis=common_axis)
                else:
                    us_low_save = np.concatenate((us_low_ex_dim, custom_mask, single_sos_channel), axis=common_axis)
                us_high_save = np.expand_dims(us_high_samples[:,:,high_channel], axis=common_axis)

                dict_us_single = {name_us_low: us_low_save,
                                  name_us_high: us_high_save}

                save_dict_with_pickle(file=dict_us_single,
                                      folder_name=save_folder + '/ultrasound', file_name=name_us_save)
    elif attention_mask == 'complex':
        suitable_couplant, index_couplant = find_suitable_index(single=couplant_sos, multiple=single_sos, threshold=16)
        if not suitable_couplant:
            # if we cannot find a suitable low quality image with couplant sos, this sample has no value in complex
            return
        else:
            single_sos_couplant = np.expand_dims(np.full(tissue_mask.shape, single_sos[index_couplant]),
                                                 axis=common_axis)
            low_single_couplant = np.expand_dims(us_low_samples[:, :, index_couplant], axis=common_axis)

        for high_channel in range(us_high_samples.shape[2]):
            suitable_dual, index_dual = find_suitable_index(single=tissue_sos[high_channel], multiple=single_sos,
                                                            threshold=32)
            if not suitable_dual:
                # if we cannot find a suitable low quality image with couplant sos, this dual sample is skipped
                continue
            else:
                single_sos_dual = np.expand_dims(np.full(tissue_mask.shape, single_sos[index_dual]),
                                                 axis=common_axis)
                low_single_dual = np.expand_dims(us_low_samples[:, :, index_dual], axis=common_axis)

                custom_mask = np.copy(tissue_mask)
                custom_mask[custom_mask == 0] = couplant_sos
                custom_mask[custom_mask == 1] = tissue_sos[high_channel]
                custom_mask = np.expand_dims(custom_mask, axis=common_axis)

                us_low_save = np.concatenate((low_single_couplant, low_single_dual, custom_mask, single_sos_couplant,
                                              single_sos_dual), axis=common_axis)

                us_high_save = np.expand_dims(us_high_samples[:, :, high_channel], axis=common_axis)

                name_us_low = 'US_low_' + study_folder + '_' + scan_num + '_ch' + str(index_couplant) + 'and' + str(
                    index_dual)
                name_us_high = 'US_high_' + study_folder + '_' + scan_num + '_ch' + str(high_channel)
                name_us_save = 'US_' + study_folder + '_' + scan_num + '_ch' + str(index_couplant) + 'and' + str(
                    index_dual) + 'to' + str(high_channel)

                dict_us_single = {name_us_low: us_low_save,
                                  name_us_high: us_high_save}

                save_dict_with_pickle(file=dict_us_single,
                                      folder_name=save_folder + '/ultrasound', file_name=name_us_save)

    else:
        for sos in range(us_high_samples.shape[2]):
            custom_mask = np.copy(tissue_mask)
            custom_mask[custom_mask == 0] = couplant_sos
            custom_mask[custom_mask == 1] = tissue_sos[sos]
            custom_mask = np.expand_dims(custom_mask, axis=common_axis)

            name_us_low = 'US_low_' + study_folder + '_' + scan_num + '_sos' + str(sos)
            name_us_high = 'US_high_' + study_folder + '_' + scan_num + '_sos' + str(sos)
            name_us_save = 'US_' + study_folder + '_' + scan_num + '_sos' + str(sos)

            us_low_save = custom_mask
            us_high_save = custom_mask

            dict_us_single = {name_us_low: us_low_save,
                              name_us_high: us_high_save}

            save_dict_with_pickle(file=dict_us_single,
                                  folder_name=save_folder + '/ultrasound', file_name=name_us_save)



# File Saving and Loading

def save_dict_with_pickle(file, folder_name, file_name):
    # use this to save pairs of low and high quality pictures
    with open(folder_name + '/' + file_name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Hetero search for suitable single speed of sounds compared to dual speed of sound


def find_suitable_index(single, multiple, threshold):
    suitable = False
    index_min = None
    if np.min(np.absolute(multiple - single)) < threshold:
        index_min = np.argmin(np.absolute(multiple - single))
        suitable = True

    return suitable, index_min


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


def do_blur(x, y, file_prefix, filename, end_folder, path_to_augment, path_to_params, data_type, attention_mask):
    aug_x, aug_y, params = data.augmentation.blur(x, y,
                                                  lower_lim=0.5, upper_lim=1.5, data_type=data_type, attention_mask=attention_mask)

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


def do_speckle_noise(x, y, file_prefix, filename, end_folder, path_to_augment, path_to_params, data_type, attention_mask):
    aug_x, aug_y, params = data.augmentation.speckle_noise(x, y, data_type=data_type, attention_mask=attention_mask)
    dict_save, name_save = create_file_names_and_dict(aug_x, aug_y, file_prefix, filename, aug_type='speckle_noise')

    save_dict_with_pickle(dict_save, path_to_augment + "/speckle_noise/" + end_folder, name_save)
    name_save_params = name_save + '_params'
    save_dict_with_pickle(params, path_to_params + '/augmentation/speckle_noise/' + end_folder, name_save_params)


def create_file_names_and_dict(aug_x, aug_y, file_prefix, filename, aug_type):
    name_low = file_prefix + '_low_' + aug_type
    name_high = file_prefix + '_high_' + aug_type
    name_save = file_prefix + '_' + filename + '_' + aug_type

    dict_save = {name_low: aug_x,
                 name_high: aug_y}
    return dict_save, name_save
