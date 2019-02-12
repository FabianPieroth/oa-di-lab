import numpy as np
import torch

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# create the zero_one tensors for the model forward


def create_zero_one_ratio(attention_anchors, attention_input_dist, attention_network_dist, shape_tensor,
                          ratio_overlap, upper_ratio, start='Not', ratio_up_to_low_channel=0.5, masks=None):
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
        if masks is None:
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
        else:
            for i in range(masks.shape[0]):
                mask = masks[i, :, :]
                single_sample_mask = mask_for_single(mask=mask, attention_network_dist=attention_network_dist,
                                                     ratio_overlap=ratio_overlap, start=start,
                                                     attention_input_dist=attention_input_dist,
                                                     shape=(shape_tensor[1], shape_tensor[2], shape_tensor[3]),
                                                     attention_anchors=attention_anchors)
                zero_one[i, :, :, :] = single_sample_mask


    zero_one = torch.tensor(zero_one)
    if torch.cuda.is_available():
        zero_one = zero_one.cuda()
    return zero_one


def mask_for_single(mask, attention_network_dist, ratio_overlap, start, attention_input_dist, shape, attention_anchors):
    single_masks = create_shape_masks(mask=mask, ratio_overlap=ratio_overlap, attention_anchors=attention_anchors,
                                      attention_input_dist=attention_input_dist, shape=shape)
    if start == 'simple':
        num_sos_masks = shape[0] - np.sum(attention_input_dist)
        intermediate_zero = np.ones((num_sos_masks, shape[1], shape[2]))
        channel = np.concatenate((single_masks, intermediate_zero), axis=0)
        # plt.imshow(single_masks[1, :, :], cmap='gray')
    else:
        network_dist = np.around(np.array(attention_network_dist) * shape[0]).astype(int)
        if not np.sum(network_dist) == shape[0]:
            index_max = np.argmax(network_dist)
            network_dist[index_max] = (network_dist[index_max] - np.sum(network_dist) + shape[0]).astype(int)
        masks_mult = np.expand_dims(single_masks[0, :, :], axis=0)
        intermediate_zero = np.repeat(masks_mult, repeats=network_dist[0], axis=0)
        for i in range(1, len(attention_anchors)):
            masks_mult = np.expand_dims(single_masks[i, :, :], axis=0)
            masks_mult = np.repeat(masks_mult, repeats=network_dist[i], axis=0)
            intermediate_zero = np.concatenate((intermediate_zero, masks_mult), axis=0)
        channel = intermediate_zero

    return channel


def crop_and_pad_sos(img, crop_int, pad_with_ones=False):
    if pad_with_ones:
        end_img = np.ones(img.shape)
    else:
        end_img = np.zeros(img.shape)
    if crop_int < 0:
        end_img[-crop_int:, :] = img[:end_img.shape[0] + crop_int, :]
    else:
        end_img[:end_img.shape[0] - crop_int, :] = img[crop_int:, :]
    return end_img


def create_shape_masks(mask, ratio_overlap, attention_anchors, attention_input_dist, shape):
    channel = np.zeros((np.sum(attention_input_dist), shape[1], shape[2]))
    mask_np = mask.detach().cpu().numpy()
    mask_np = 1 - (mask_np - np.min(mask_np)) / np.max(mask_np - np.min(mask_np))
    mask_1 = crop_and_pad_sos(img=mask_np, crop_int=8)
    rows_top, rows_bottom = row_top_and_bottom(mask=mask_1)

    anchors = attention_anchors[1:] / np.sum(attention_anchors[1:])

    num_anchors = [rows_bottom] + list(np.around((mask.shape[0] - rows_bottom) * anchors).astype(int))
    if not np.sum(num_anchors) == mask.shape[0]:
        num_anchors[np.argmax(num_anchors)] = (num_anchors[np.argmax(num_anchors)] - np.sum(num_anchors) +
                                               mask.shape[0]).astype(int)
    for i in range(1, len(num_anchors)):
        shifted_mask = shift_preserve_shape(width=num_anchors[i], start_mask=mask_1,
                                            end_point=np.sum(num_anchors[:i]) - num_anchors[0])
        shifted_mask = expand_by_ratio(mask=shifted_mask, ratio=ratio_overlap)
        shifted_mask = shrink_masks(mask=shifted_mask, target_shape=(shape[1], shape[2]))
        channel[i, :, :] = shifted_mask
    mask_1 = expand_by_ratio(mask=mask_1, ratio=ratio_overlap)
    mask_1 = shrink_masks(mask=mask_1, target_shape=(shape[1], shape[2]))
    channel[0, :, :] = mask_1
    return channel


def shrink_masks(mask, target_shape):
    num_rows, num_cols = mask.shape
    num_rows_del = num_rows - target_shape[0]
    num_cols_del = num_cols - target_shape[1]
    rows_to_del = np.linspace(start=0, stop=mask.shape[0] - 1, num=num_rows_del).astype(int)
    cols_to_del = np.linspace(start=0, stop=mask.shape[1] - 1, num=num_cols_del).astype(int)
    mask = np.delete(mask, list(rows_to_del), axis=0)
    mask = np.delete(mask, list(cols_to_del), axis=1)
    return mask


def shift_preserve_shape(width, start_mask, end_point):
    mask_pad = crop_and_pad_sos(start_mask, crop_int=-width, pad_with_ones=True)
    return_mask = np.logical_and(1 - start_mask, mask_pad).astype(int)
    return_mask = crop_and_pad_sos(return_mask, crop_int=-end_point, pad_with_ones=False)
    return return_mask


def expand_by_ratio(mask, ratio):

    return_mask = np.zeros(mask.shape)
    rows_top, rows_bottom = row_top_and_bottom(mask=mask, left_and_middle=True)
    middle_row = int((rows_top + rows_bottom) / 2)
    extend_top = int(rows_top * ratio)
    extend_bottom = int((mask.shape[0] - rows_bottom) * ratio)
    return_mask[:middle_row - extend_top, :] = mask[extend_top: middle_row, :]
    return_mask[middle_row - extend_top:middle_row, :] = np.ones((extend_top, mask.shape[1]))
    return_mask[middle_row:middle_row + extend_bottom, :] = np.ones((extend_bottom, mask.shape[1]))
    return_mask[middle_row + extend_bottom:, :] = mask[middle_row:mask.shape[0] - extend_bottom, :]

    return return_mask


def row_top_and_bottom(mask, left_and_middle=False):
    if left_and_middle:
        rows, _ = np.where(mask[:, 0:2] == 1)
        top = np.min(rows)
        rows_trunc, _ = np.where(mask[top:, :] == 0)
        bottom = np.min(rows_trunc) + top
    else:
        rows, _ = np.where(mask == 1)
        top, bottom = np.min(rows), np.max(rows)
    return top, bottom


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
