def scale_image(image, min_data, max_data, image_type, min_out=-1, max_out=1):
    if image_type == 'US':
        factor = (max_out-min_out)/(max_data - min_data)
        additive = min_out - min_data*(max_out-min_out)/(max_data - min_data)
        image_out = image*factor + additive
    else:
        print('scale and center not implemented yet for OA data')
    return image_out

def scale_batch(batch, min_data, max_data, image_type, min_out=-1, max_out=1):
    if image_type == 'US':
        factor = (max_out-min_out)/(max_data - min_data)
        additive = min_out - min_data*(max_out-min_out)/(max_data - min_data)
        batch_out = batch*factor + additive
    else: print('scale and center not implemented yet for OA data')
    return batch_out

def scale_and_center(batch, scale_params, mean_image, image_type):
    if image_type == 'US':
        [min_data, max_data] = scale_params
        mean_scaled = scale_image(mean_image, min_data, max_data, image_type = image_type)
        batch_out = scale_batch(batch, min_data, max_data)
        batch_out = image_out - mean_scaled
    else: print('scale and center not implemented yet for OA data')
    return batch_out

def scale_and_center_reverse(batch, scale_params, mean_image):
    [min_data, max_data] = scale_params
    mean_scaled = scale_image(mean_image, min_data, max_data)
    batch_out = batch + mean_scaled
    batch_out = scale_batch(batch_out, -1,1,min_data, max_data)
    return batch_out