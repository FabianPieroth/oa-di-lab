from pathlib import Path
import pickle


def scale_image(image, min_data, max_data, image_type, min_out=-1, max_out=1):
    """ scales the input image from [min_data,max_data] to [min_out, max_out]
        input: image: for US (H,W) array, for OA: (H,W,C) array
               min_data, max_data: minimum and maximum over the data set (channel by channel)
                    for US: floats, for OA: arrays of shape (C,)
               image_type: 'US' or 'OA'
        output: image_out: array with same shape as image
    """

    factor = (max_out - min_out) / (max_data - min_data)
    additive = min_out - min_data * (max_out - min_out) / (max_data - min_data)
    if image_type == 'US':
        image_out = image * factor + additive
    else:
        image_out = image * factor[None, None, :] + additive[None, None, :]
    return image_out


def scale_batch(batch, min_data, max_data, image_type, min_out=-1, max_out=1):
    """ scales the input batch from [min_data,max_data] to [min_out, max_out]
            input: batch: for US (N,H,W) array, for OA: (N,H,W,C) array
                   min_data, max_data: minimum and maximum over the data set (channel by channel)
                        for US: floats, for OA: arrays of shape (C,)
                   image_type: 'US' or 'OA'
            output: batch_out array with same shape as batch
    """
    factor = (max_out - min_out) / (max_data - min_data)
    additive = min_out - min_data * (max_out - min_out) / (max_data - min_data)
    if image_type == 'US':
        batch_out = batch * factor + additive
    else:
        batch_out = batch * factor[None, None, None, :] + additive[None, None, None, :]
    return batch_out


def scale_and_center(batch, scale_params, mean_image, image_type):
    """ scales and centers the input batch given the scale_params and the mean_image
            input: batch: for US (N,H,W) array, for OA: (N,H,W,C) array
                   scale_params: minimum and maximum over the data set (channel by channel)
                        for US: array of shape (2,), for OA: arrays of shape (2,C)
                    mean_image: mean image for the whole data set
                        for US: array of shape (H,W), for OA: (H,W,C)
                   image_type: 'US' or 'OA'
            output: batch_out array with same shape as batch"""
    [min_data, max_data] = scale_params
    mean_scaled = scale_image(image=mean_image, min_data=min_data, max_data=max_data, image_type=image_type)
    batch_out = scale_batch(batch=batch, min_data=min_data, max_data=max_data, image_type=image_type)
    batch_out = batch_out - mean_scaled
    return batch_out


def scale_and_center_reverse(batch, scale_params, mean_image, image_type):
    """ undoes the scaling and centering of the input batch given the scale_params and the mean_image
                input: batch: for US (N,H,W) array, for OA: (N,H,W,C) array
                       scale_params: minimum and maximum over the data set (channel by channel)
                            for US: array of shape (2,), for OA: arrays of shape (2,C)
                        mean_image: mean image for the whole data set
                            for US: array of shape (H,W), for OA: (H,W,C)
                       image_type: 'US' or 'OA'
                output: batch_out array with same shape as batch"""
    [min_data, max_data] = scale_params
    mean_scaled = scale_image(image=mean_image, min_data=min_data, max_data=max_data, image_type=image_type)
    batch_out = batch + mean_scaled
    batch_out = scale_batch(batch_out, min_data=-1, max_data=1, image_type=image_type, min_out=min_data,
                            max_out=max_data)
    return batch_out


def load_params(image_type, param_type):
    """ loads the specified parameters from file
        input: image_type: 'US' or 'OA'
                param_type: 'scale' or 'mean_image' (maybe more options later)
        output: params_low, params_high: the parameters."""
    #dir_params = '/mnt/local/mounted'
    dir_params = str(Path().resolve().parents[1] / 'params')
    if param_type in ['scale_params', 'mean_images']:
        file_name = image_type + '_' + param_type
        filepath = dir_params + '/' + 'scale_and_center' + '/' + file_name
        with open(filepath, 'rb') as handle:
            params = pickle.load(handle)
    else: print('invalid parameter type')
    dic_key_low = image_type + '_low'
    dic_key_high = image_type + '_high'
    params_low = params[dic_key_low]
    params_high = params[dic_key_high]
    return params_low, params_high