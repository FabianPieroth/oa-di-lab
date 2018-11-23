import pickle
import data.augmentation


# this File contains several helper functions

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


def create_file_names_and_dict(aug_x, aug_y, file_prefix, filename, aug_type):
    name_low = file_prefix + '_low_' + aug_type
    name_high = file_prefix + '_high_' + aug_type
    name_save = file_prefix + '_' + filename + '_' + aug_type

    dict_save = {name_low: aug_x,
                 name_high: aug_y}
    return dict_save, name_save
