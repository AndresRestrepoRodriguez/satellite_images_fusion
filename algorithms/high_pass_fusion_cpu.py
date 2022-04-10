import numpy as np
import skimage
from scipy import ndimage

initial_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * (1 / 9)
second_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * (1 / 9)


def fusion_high_pass(band, image_a, image_b):
    band_fusion = band + np.multiply(np.divide(band, image_b), image_a)
    return band_fusion


def fusion_high_pass_cpu(multispectral_image, panchromatic_image):
    union_list = []
    num_bands = int(multispectral_image.shape[2])
    panchormatic_float = panchromatic_image.astype(np.float32)
    image_initial_filter = ndimage.correlate(panchormatic_float, initial_filter, mode='constant')
    image_second_filter = ndimage.correlate(panchormatic_float, second_filter, mode='constant')
    image_initial_filter[image_initial_filter < 0] = 0
    image_second_filter[image_second_filter < 0] = 0
    image_initial_filter_float = image_initial_filter.astype(np.float32)
    image_second_filter_float = image_second_filter.astype(np.float32)
    band_iterator = 0
    while band_iterator < num_bands:
        band = multispectral_image[:, :, band_iterator]
        band_float = band.astype(np.float32)
        fusion_bands = fusion_high_pass(band_float, image_initial_filter_float, image_second_filter_float)
        fusion_bands[fusion_bands > 255] = 255
        result_image = fusion_bands.astype(np.uint8)
        union_list.append(result_image)
        band_iterator = band_iterator + 1
    fusioned_image = np.stack(union_list,axis = 2)
    return fusioned_image
