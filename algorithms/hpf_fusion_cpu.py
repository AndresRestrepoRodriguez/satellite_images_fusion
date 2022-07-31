import numpy as np
from scipy import ndimage


def fusion_hpf_cpu(multispectral_image, panchromatic_image):
    float_list = []
    band_iterator = 0
    panchromatic_float = panchromatic_image.astype(np.float32)
    num_bands = int(multispectral_image.shape[2])

    while band_iterator < num_bands:
        band = multispectral_image[:, :, band_iterator]
        band_float = band.astype(np.float32)
        float_list.append(band_float)
        band_iterator = band_iterator + 1
    image = create_filter(panchromatic_float)
    variance = get_variance(multispectral_image, num_bands)
    final_image = merge_image(panchromatic_image, variance, float_list, num_bands, image)
    return final_image


def create_filter(float_panchromatic):
    filter = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, 80, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1]]) * (1 / 106);

    image = ndimage.correlate(float_panchromatic, filter, mode='constant').astype(np.int8)
    image[image < 0] = 0
    return image


def get_variance(multispectral_image, num_bands):
    zeros_matrix = np.zeros((num_bands, 2))
    for n in range(num_bands):
        zeros_matrix[n][0] = n
        variance = np.std(multispectral_image[:, :, n])
        zeros_matrix[n][1] = variance
    return zeros_matrix


def merge_image(panchromatic_image, zeros_matrix, float_list, num_bands, image_data):
    std_panchromatic = np.std(panchromatic_image)
    sum_variance = 0
    for i in zeros_matrix:
        sum_variance += i[1]
    total_var = sum_variance / (std_panchromatic * 0.65)
    bands_list = []
    for j in range(num_bands):
        base = float_list[j] + image_data * total_var
        base[base > 255] = 255
        base[base < 0] = 0
        base = base.astype(np.uint8)
        bands_list.append(base)
    fusioned_image = np.stack(bands_list, axis=2)
    return fusioned_image


