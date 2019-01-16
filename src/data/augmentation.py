import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate as ipol
import re
import scipy.io
import os

#from skimage.transform import resize

import random



##### Elastic Deformation #######

# deform function
def elastic_deform_helper(image, x_coord, y_coord, dx, dy):
    """ Applies random elastic deformation to the input image
        with given coordinates and displacement values of deformation points.
        Keeps the edge of the image steady by adding a few frame points that get displacement value zero.
    Input: image: array of shape (N.M,C) (Haven't tried it out for N != M), C number of channels
           x_coord: array of shape (L,) contains the x coordinates for the deformation points
           y_coord: array of shape (L,) contains the y coordinates for the deformation points
           dx: array of shape (L,) contains the displacement values in x direction
           dy: array of shape (L,) contains the displacement values in x direction
    Output: the deformed image (shape (N,M,C))
    """

    # Preliminaries
    # dimensions of the input image
    shape = image.shape

    # centers of x and y axis
    x_center = shape[1] / 2
    y_center = shape[0] / 2

    ## Construction of the coarse grid

    # anker points: coordinates
    x_coord_anker_points = np.array([0, x_center, shape[1] - 1, 0, shape[1] - 1, 0, x_center, shape[1] - 1])
    y_coord_anker_points = np.array([0, 0, 0, y_center, y_center, shape[0] - 1, shape[0] - 1, shape[0] - 1])
    # anker points: values
    dx_anker_points = np.zeros(8)
    dy_anker_points = np.zeros(8)

    # combine deformation and anker points to coarse grid
    x_coord_coarse = np.append(x_coord, x_coord_anker_points)
    y_coord_coarse = np.append(y_coord, y_coord_anker_points)
    coord_coarse = np.array(list(zip(y_coord_coarse, x_coord_coarse)))

    dx_coarse = np.append(dx, dx_anker_points)
    dy_coarse = np.append(dy, dy_anker_points)

    ## Interpolation onto fine grid
    # coordinates of fine grid
    coord_fine = [[y, x] for y in range(shape[0]) for x in range(shape[1])]
    # interpolate displacement in both x and y direction
    dx_fine = ipol.griddata(coord_coarse, dx_coarse, coord_fine,
                            method='cubic')  # cubic works better but takes longer (?)
    dy_fine = ipol.griddata(coord_coarse, dy_coarse, coord_fine, method='cubic')  # other options: 'linear'
    # get the displacements into shape of the input image (the same values in each channel)


    if len(shape) == 3:
        dx_fine = dx_fine.reshape(shape[0:2])
        dx_fine = np.stack([dx_fine] * shape[2], axis=-1)
        dy_fine = dy_fine.reshape(shape[0:2])
        dy_fine = np.stack([dy_fine] * shape[2], axis=-1)

        ## Deforming the image: apply the displacement grid
        # base grid
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        # add displacement to base grid (-> new coordinates)
        indices = np.reshape(y + dy_fine, (-1, 1)), np.reshape(x + dx_fine, (-1, 1)), np.reshape(z, (-1, 1))

    else:
        dx_fine = dx_fine.reshape(shape)
        dy_fine = dy_fine.reshape(shape)
        ## Deforming the image: apply the displacement grid
        # base grid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        # add displacement to base grid (-> new coordinates)
        indices = np.reshape(y + dy_fine, (-1, 1)), np.reshape(x + dx_fine, (-1, 1))
    # evaluate the image at the new coordinates
    deformed_image = map_coordinates(image, indices, order=2, mode='nearest')
    deformed_image = deformed_image.reshape(image.shape)

    return deformed_image


# wrapper function
def elastic_deform(image1, image2, n_points=1, stdev_displacement_fac=0.05, deformation_points_location_fac=0.7):
    """ Generates random points and displacements and performs elastic deformation
        on the two input images with these deformation points
        Input: image1, image2: arrays of shape (N,M,C), C being the number of channels
               n_points: number of deformation points
               stdev_displacement_fac: factor that is multiplied with the smaller x or y dimension
                            of the input image to get the standard deviation of the discplacement values
               deformation_points_location_fac: value in (0,1), gives size of box in which the random points are generated
        Output: the deformed images (shape (N,M,C))
               """
    ## Preliminaries
    # dimensions of the input images
    shape = image1.shape

    # centers of x and y axis
    x_center = shape[1] / 2
    y_center = shape[0] / 2

    ## deformation points
    # coordinates
    left_limit = int(x_center - deformation_points_location_fac * x_center)
    right_limit = int(x_center + deformation_points_location_fac * x_center)
    upper_limit = int(y_center - deformation_points_location_fac * y_center)
    lower_limit = int(y_center + deformation_points_location_fac * y_center)
    x_coord = np.random.randint(left_limit, right_limit,
                                n_points)  # from left_limit (inclusive) to right_limit (exclusive)
    y_coord = np.random.randint(upper_limit, lower_limit, n_points)

    # displacement values
    stdev_displacement = stdev_displacement_fac*min(shape[0], shape[1])
    dx = np.random.randn(n_points) * stdev_displacement
    dy = np.random.randn(n_points) * stdev_displacement

    ## perform the elastic deformation
    deformed_image1 = elastic_deform_helper(image1, x_coord, y_coord, dx, dy)
    deformed_image2 = elastic_deform_helper(image2, x_coord, y_coord, dx, dy)

    # prepare parameters for output
    params = [x_coord, y_coord, dx, dy]

    return deformed_image1, deformed_image2, params

##### Crop and stretch #####

def crop_stretch_helper(in_image, side, crop_size):
    #helper function to crop both the input and target image by some crop size
    # TODO: delete unneccessary stuff
    if side == 1:
        cropped = in_image[:in_image.shape[0] - crop_size, :in_image.shape[1] - crop_size]

        resized = resize(cropped, (in_image.shape),
                         anti_aliasing=True)
    if side == 2:
        cropped = in_image[:in_image.shape[0] - crop_size, crop_size:in_image.shape[1]]

        resized = resize(cropped, (in_image.shape),
                         anti_aliasing=True)

    return resized


def crop_stretch(in_image1, in_image2):
    size = random.randint(25, 40)
    side = random.randint(1, 2)
    aug_img = crop_stretch_helper(in_image1, side, size)
    aug_label = crop_stretch_helper(in_image2, side, size)

    params = [side, size]

    return aug_img, aug_label, params


####### Blurring ############

def blur_helper(image, sigma = 2):
    """ blurs the input image by applying a gaussian filter with the specified sigma (only use for US data)
        input: image: array of (N,N)
               sigma: int specifying gaussian filter
        output: transformed_image"""
    shape = image.shape
    if len(shape) == 2:
        transformed_image = gaussian_filter(image, sigma = sigma)
    else:
        transformed_image = np.empty_like(image)
        for channel in range(shape[2]):
            transformed_image[:,:,channel] = gaussian_filter(image[:,:,channel], sigma = sigma)
    return transformed_image


def blur(image1, image2, lower_lim = 0.2, upper_lim = 1.5, data_type='homo', attention_mask='Not'):
    """ blurs image1 with a blur_helper, only use for US data
        input: image1: the image to be blurred
               image2: the target image, not to be blurred
               lower_lim, upper_lim: ints lower and upper lim for the sigma for the gaussian filter
        output: transformed_image1, image2"""
    sig = np.random.random(1)[0]*(upper_lim - lower_lim) + lower_lim
    if data_type == 'hetero':
        if attention_mask == 'Not':
            input_image1 = image1[:, :, 0]
            transformed_image1_temp = blur_helper(input_image1, sigma=sig)
            image1[:, :, 0] = transformed_image1_temp
            transformed_image1 = image1
        elif attention_mask == 'simple':
            input_image1a = image1[:, :, 0]
            input_image1b = image1[:, :, 1]
            transformed_image1a_temp = blur_helper(input_image1a, sigma=sig)
            transformed_image1b_temp = blur_helper(input_image1b, sigma=sig)
            image1[:, :, 0] = transformed_image1a_temp
            image1[:, :, 1] = transformed_image1b_temp
            transformed_image1 = image1
    else:
        transformed_image1 = blur_helper(image1, sigma=sig)

    return transformed_image1, image2, sig


########### Flip ####################
def flip(image1,image2):
    #flips boths images along the vertical axis

    return np.flip(image1,axis=1), np.flip(image2,axis=1)

# Load random channels from raw data


def _extract_info_from_filename(filename):
    # extract image_type, new_in folder, Scan number and channel
    list_of_infos = filename.split('_')
    image_type = list_of_infos[0]
    read_folder = list_of_infos[1]
    scan = list_of_infos[2] + '_' + list_of_infos[3]
    channel = int(re.search(r'\d+', list_of_infos[4]).group())

    return image_type, read_folder, scan, channel

def rchannels(filename, dir_raw_in, num_rchannels=2):
    image_type, read_folder, scan, channel = _extract_info_from_filename(filename)

    in_dir = dir_raw_in + '/' + read_folder + '/' + scan
    in_files = os.listdir(in_dir)
    raw_file_name = [s for s in in_files if image_type + '_' in s][0]
    raw_file = scipy.io.loadmat(in_dir + '/' + raw_file_name)
    range_to_sample_from = np.delete(np.array(range(raw_file[image_type + '_low'].shape[2])), [channel])
    sample_number = min(num_rchannels, raw_file[image_type + '_low'].shape[2] - 1)
    if sample_number < num_rchannels:
        print("Chosen number of channels for augmented is higher than some maximum number of channels.")
    sampled_channels = np.random.choice(a=range_to_sample_from, size=sample_number, replace=False)
    ret_list = []
    ret_save_names = []
    for i in sampled_channels:
        name_low = image_type + '_low_' + read_folder + '_' + scan + '_ch' + str(i)
        name_high = image_type + '_high_' + read_folder + '_' + scan + '_ch' + str(i)
        name_save = image_type + '_' + read_folder + '_' + scan + '_ch' + str(i)
        dict_single = {name_low: raw_file[image_type + '_low'][:, :, i],
                       name_high: raw_file[image_type + '_high'][:, :, i]}
        ret_save_names.append(name_save)
        ret_list.append(dict_single)

    return ret_list, ret_save_names

########### Speckle Noise ####################


def speckle_noise_helper(image, mult_noise):
    """ """
    transformed_image = image * mult_noise
    return transformed_image


def speckle_noise(image1, image2, lower_lim_stdev=0.1, upper_lim_stdev=0.15, data_type='homo', attention_mask='Not'):
    """ """
    stdev = np.random.random(1)*(upper_lim_stdev - lower_lim_stdev) + lower_lim_stdev
    shape = image1.shape
    dim = shape[0] * shape[1]
    eta = np.random.randn(dim) * stdev + 1
    eta = eta.reshape(shape[:2])
    if data_type == 'hetero':
        if attention_mask == 'Not':
            input_image1 = image1[:, :, 0]
            transformed_image1_temp = speckle_noise_helper(input_image1, mult_noise=eta)
            image1[:, :, 0] = transformed_image1_temp
            transformed_image1 = image1
        elif attention_mask == 'simple':
            input_image1a = image1[:, :, 0]
            input_image1b = image1[:, :, 1]
            transformed_image1a_temp = speckle_noise_helper(input_image1a, mult_noise=eta)
            transformed_image1b_temp = speckle_noise_helper(input_image1b, mult_noise=eta)
            image1[:, :, 0] = transformed_image1a_temp
            image1[:, :, 1] = transformed_image1b_temp
            transformed_image1 = image1
    else:
        transformed_image1 = speckle_noise_helper(image1, mult_noise=eta)
    return transformed_image1, image2, eta


