import numpy as np
import torch

# create the zero_one tensors for the model forward


def create_zero_one_ratio(attention_anchors, attention_input_dist, attention_network_dist, shape_tensor,
                          ratio_overlap, upper_ratio, start='Not', ratio_up_to_low_channel=0.5):
    # shape_tensor: (N, C, H, W)
    zero_one = np.zeros(shape_tensor)
    version = 'new'
    if version == 'old':
        lower_ratio = 1 - upper_ratio + (upper_ratio) * ratio_overlap
        upper_ratio = upper_ratio + (1 - upper_ratio) * ratio_overlap
        num_upper = int(shape_tensor[2] * upper_ratio)
        num_lower = shape_tensor[2] - int(shape_tensor[2] * lower_ratio)
        single_upper = np.zeros((shape_tensor[2], shape_tensor[3]))
        single_lower = np.ones((shape_tensor[2], shape_tensor[3]))
        single_upper[:num_upper, :] = 1.0
        single_lower[:num_lower, :] = 0.0
        if start == 'simple':
            for i in range(shape_tensor[0]):
                zero_one[i, 0, :, :] = single_upper
                zero_one[i, 1, :, :] = single_lower
                zero_one[i, 2, :, :] = np.ones((shape_tensor[2], shape_tensor[3]))
                zero_one[i, 3, :, :] = np.ones((shape_tensor[2], shape_tensor[3]))
        elif start == 'complex':
            for i in range(shape_tensor[0]):
                zero_one[i, 0, :, :] = single_upper
                zero_one[i, 1, :, :] = single_lower
                zero_one[i, 2, :, :] = np.ones((shape_tensor[2], shape_tensor[3]))
                zero_one[i, 3, :, :] = np.ones((shape_tensor[2], shape_tensor[3]))
                zero_one[i, 4, :, :] = np.ones((shape_tensor[2], shape_tensor[3]))
        else:
            num_channel_till_up = int(shape_tensor[1] * ratio_up_to_low_channel)
            for i in range(shape_tensor[0]):
                for j in range(shape_tensor[1]):
                    if j < num_channel_till_up:
                        zero_one[i, j, :, :] = single_upper
                    else:
                        zero_one[i, j, :, :] = single_lower
    else:
        num_anchors = np.around(np.array(attention_anchors) * shape_tensor[2]).astype(int)
        if not np.sum(num_anchors) == shape_tensor[2]:
            num_anchors[np.argmax(num_anchors)] = (num_anchors[np.argmax(num_anchors)] - np.sum(num_anchors) +
                                                   shape_tensor[2]).astype(int)
        if start == 'simple':
            num_sos_masks = shape_tensor[1] - np.sum(attention_input_dist)
            intermediate_zero = np.ones((num_sos_masks, shape_tensor[2], shape_tensor[3]))
            for i in range(len(attention_anchors) - 1, -1, -1):
                channel = create_single_mask(shape_image=(shape_tensor[2], shape_tensor[3]),
                                             start=np.sum(num_anchors[:i + 1]) - num_anchors[i],
                                             width=num_anchors[i], increase_ratio=ratio_overlap)
                channel = np.expand_dims(channel, axis=0)
                intermediate_zero = np.concatenate((channel, intermediate_zero), axis=0)
            zero_one = np.repeat(np.expand_dims(intermediate_zero, axis=0), repeats=shape_tensor[0], axis=0)
        else:
            network_dist = np.around(np.array(attention_network_dist) * shape_tensor[1]).astype(int)
            if not np.sum(network_dist) == shape_tensor[1]:
                index_max = np.argmax(network_dist)
                network_dist[index_max] = (network_dist[index_max] - np.sum(network_dist) +
                                           shape_tensor[1]).astype(int)
            channel = create_single_mask(shape_image=(shape_tensor[2], shape_tensor[3]),
                                         start=0, width=num_anchors[0], increase_ratio=ratio_overlap)
            channel = np.expand_dims(channel, axis=0)
            intermediate_zero = np.repeat(channel, repeats=network_dist[0], axis=0)
            for i in range(1, len(attention_anchors)):
                channel = create_single_mask(shape_image=(shape_tensor[2], shape_tensor[3]),
                                             start=np.sum(num_anchors[:i + 1]) - num_anchors[i],
                                             width=num_anchors[i], increase_ratio=ratio_overlap)
                channel = np.expand_dims(channel, axis=0)
                channel = np.repeat(channel, repeats=network_dist[i], axis=0)
                intermediate_zero = np.concatenate((intermediate_zero, channel), axis=0)
            zero_one = np.repeat(np.expand_dims(intermediate_zero, axis=0), repeats=shape_tensor[0], axis=0)

    zero_one = torch.tensor(zero_one)
    if torch.cuda.is_available():
        zero_one = zero_one.cuda()
    return zero_one


def create_single_mask(shape_image, start, width, increase_ratio):
    channel = np.zeros(shape_image)
    side_length = shape_image[0] - 1
    if start + width > side_length + 1:
        print('There was something wrong in the function call for the zero_one matrices!')
        return channel
    upper = int(np.ceil(start * increase_ratio))
    lower = int(np.ceil((side_length - start - width) * increase_ratio))
    channel[(start - upper):start + width + lower + 1, :] = 1.0

    return channel
