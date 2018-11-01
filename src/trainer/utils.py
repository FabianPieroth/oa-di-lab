def scale_image(image, min_data, max_data, min_out=-1, max_out=1):
    """ Scales the image from range [min_data, max_data] to [min_out,max_out]
        Now only for 2dim data (US)"""
    factor = (max_out-min_out)/(max_data - min_data)
    additive = min_out - min_data*(max_out-min_out)/(max_data - min_data)
    image_out = image*factor + additive
    return image_out

def scale_image_reverse(scaledimage, min_data, max_data, min_in= -1, max_in = 1):
    """ Reverses the scaling, transforms the image back to [min_data, max_data]
        Now only for 2dim data (US)"""
    image_out = scale_image(scaledimage, min_in, max_in, min_data, max_data)
    return image_out