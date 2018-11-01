git import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy import interpolate as ipol
from skimage.transform import rescale, resize, downscale_local_mean

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

    ## Preliminaries
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
    coord_coarse = np.array(list(zip(x_coord_coarse, y_coord_coarse)))

    dx_coarse = np.append(dx, dx_anker_points)
    dy_coarse = np.append(dy, dy_anker_points)

    ## Interpolation onto fine grid
    # coordinates of fine grid
    coord_fine = [[x, y] for x in range(shape[1]) for y in range(shape[0])]
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
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        # add displacement to base grid (-> new coordinates)
        indices = np.reshape(y + dy_fine, (-1, 1)), np.reshape(x + dx_fine, (-1, 1)), np.reshape(z, (-1, 1))
    else:
        dx_fine = dx_fine.reshape(shape)
        dy_fine = dy_fine.reshape(shape)
        ## Deforming the image: apply the displacement grid
        # base grid
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        # add displacement to base grid (-> new coordinates)
        indices = np.reshape(y + dy_fine, (-1, 1)), np.reshape(x + dx_fine, (-1, 1))
    # evaluate the image at the new coordinates
    deformed_image = map_coordinates(image, indices, order=2, mode='nearest')
    deformed_image = deformed_image.reshape(image.shape)

    return deformed_image


# wrapper function
def elastic_deform(image1, image2, n_points=1, stdev_displacement=20, deformation_points_location_fac=0.5):
    """ Generates random points and displacements and performs elastic deformation
        on the two input images with these deformation points
        Input: image1, image2: arrays of shape (N,M,C), C being the number of channels
               n_points: number of deformation points
               stdev_displacement: standard deviation of the discplacement values
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
    dx = np.random.randn(n_points) * stdev_displacement
    dy = np.random.randn(n_points) * stdev_displacement

    ## perform the elastic deformation
    deformed_image1 = elastic_deform_helper(image1, x_coord, y_coord, dx, dy)
    deformed_image2 = elastic_deform_helper(image2, x_coord, y_coord, dx, dy)

    return deformed_image1, deformed_image2


def crop_stretch_helper(in_image, side, crop_size):
    if side == 1:
        cropped = in_image[crop_size:in_image.shape[0], :in_image.shape[1] - crop_size]

        resized = resize(cropped, (in_image.shape),
                         anti_aliasing=True)
    if side == 2:
        cropped = in_image[crop_size:in_image.shape[0], crop_size:in_image.shape[1]]

        resized = resize(cropped, (in_image.shape),
                         anti_aliasing=True)

    return resized


def crop_stretch(in_image1, in_image2, rseed):
    random.seed(rseed)
    size = random.randint(25, 40)
    side = random.randint(1, 2)
    aug_img = crop_stretch_helper(in_image1, side, size)
    aug_label = crop_stretch_helper(in_image2, side, size)

    return aug_img, aug_label

