import pycuda.autoinit
import numpy as np
import skimage.io
from scipy import ndimage
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray


def bands_operation_gpu(color_band, panchromatic_data):
    color_band_gpu = gpuarray.to_gpu(color_band)
    panchromatic_data_gpu = gpuarray.to_gpu(panchromatic_data)
    result_operation = (color_band_gpu + panchromatic_data_gpu) * 0.5
    return result_operation.get()


def fusion_mean_value_gpu(multispectral_image, panchromatic_image):
    union_list = []
    num_bands = int(multispectral_image.shape[2])
    panchromatic_float = panchromatic_image.astype(np.float32)
    band_iterator = 0
    while band_iterator < num_bands:
        matrix = multispectral_image[:, :, band_iterator]
        float_matrix = matrix.astype(np.float32)
        fusion_bands = bands_operation_gpu(float_matrix, panchromatic_float)
        union = fusion_bands.astype(np.uint8)
        union_list.append(union)
        band_iterator = band_iterator + 1
    fusioned_image = np.stack(union_list, axis=2)
    return fusioned_image
