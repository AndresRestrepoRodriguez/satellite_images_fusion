import pycuda.autoinit
import numpy as np
import skimage.io
from scipy import ndimage
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import skcuda.misc as misc
import skimage
from pycuda.elementwise import ElementwiseKernel
from cupyx.scipy.ndimage import filters
import cupy as cp


def fusion_hpf_gpu(multispectral_image, panchromatic_image):
    float_list = []
    band_iterator = 0
    panchromatic_float = panchromatic_image.astype(np.float32)
    num_bands = int(multispectral_image.shape[2])

    while band_iterator < num_bands:
        band = multispectral_image[:, :, band_iterator]
        float_band = band.astype(np.float32)
        float_list.append(float_band)
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

    panchromatic_cupy = cp.array(float_panchromatic)
    filter_cupy = cp.array(filter)
    image_cupy = filters.correlate(panchromatic_cupy, filter_cupy, mode='constant')
    image_cpu = image_cupy.get()
    fitted_image = fit_negative_values(image_cpu)
    return fitted_image


def fit_negative_values(matrix):
    matrix_gpu = gpuarray.to_gpu(matrix)
    new_matrix_gpu = gpuarray.empty_like(matrix_gpu)
    fit_negative = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] < 0){z[i] = 0.0;}else{z[i] = x[i];}",
        "adjust_value")
    fit_negative(matrix_gpu, new_matrix_gpu)
    return new_matrix_gpu.get()


def get_variance(multispectral_image, num_bands):
    zeros_matrix = np.zeros((num_bands, 2))
    for n in range(num_bands):
        zeros_matrix[n][0] = n
        matrix_gpu = gpuarray.to_gpu(multispectral_image[:, :, n].astype(np.float32))
        variance_gpu = misc.std(matrix_gpu)
        zeros_matrix[n][1] = variance_gpu.get()
    return zeros_matrix


def merge_image(multispectral_image, zeros_matrix, float_list, num_bands, image_data):
    panchromatic_gpu = gpuarray.to_gpu(multispectral_image.astype(np.float32))
    std_panchromatic_gpu = misc.std(panchromatic_gpu)
    sum_variance = 0
    for i in zeros_matrix:
        sum_variance += i[1]
    total_var = sum_variance / (std_panchromatic_gpu * 0.65)
    bands_list = []
    image_gpu = gpuarray.to_gpu(image_data)
    for j in range(num_bands):
        list_gpu = gpuarray.to_gpu(float_list[j])
        multi = image_gpu * total_var.get()
        base_temp = list_gpu + multi
        greater_fitted_values = fit_greater_values(base_temp)
        float_image = greater_fitted_values.astype(np.float32)
        negative_fitted_values = fit_negative_values(float_image)
        float_image = negative_fitted_values.astype(np.float32)
        base = float_image.astype(np.uint8)
        bands_list.append(base)
    fusioned_image = np.stack(bands_list, axis=2)
    return fusioned_image


def fit_greater_values(matrix):
    matrix_gpu = matrix.astype(np.float32)
    matrix_gpu_new = gpuarray.empty_like(matrix_gpu)
    fit_positive = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] > 255){z[i] = 255.0;}else{z[i] = x[i];}",
        "adjust_value")
    fit_positive(matrix_gpu, matrix_gpu_new)

    return matrix_gpu_new.get()
