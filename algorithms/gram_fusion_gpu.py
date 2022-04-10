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

results_operations = []


def fusion_gram_gpu(multispectral_image, panchromatic_image):
    panchromatic_float = panchromatic_image.astype(np.float32)
    panchromatic_copy = panchromatic_float
    num_bands = int(multispectral_image.shape[2])
    band_iterator = 0
    bands_list = []

    while band_iterator < num_bands:
        band = multispectral_image[:, :, band_iterator]
        float_band = band.astype(np.float32)
        panchromatic_copy = panchromatic_copy + float_band
        float_band = float_band.astype(np.uint8)
        bands_list.append(float_band)
        band_iterator = band_iterator + 1
    total_bands = num_bands + 1
    fusion_mean = panchromatic_copy / total_bands
    bands_list.insert(0, fusion_mean)
    fusioned_image = np.stack(bands_list, axis=2)
    mat_escalar = get_scalar(fusioned_image)
    bands_matrix = create_bands(mat_escalar, panchromatic_float)
    fusion_bands = merge_bands(bands_matrix)
    final_image = merge_image(fusion_bands)
    return final_image


def get_scalar(fusioned_image):
    global results_operations
    fusioned_float = fusioned_image.astype(np.float32)
    matrix_tmp_gpu = gpuarray.to_gpu(fusioned_float)
    num_bands = int(fusioned_image.shape[2])
    matrix_tmp = np.empty_like(fusioned_image)

    for band_iterator in range(num_bands):
        matrix_tmp[:, :, band_iterator] = fusioned_image[:, :, band_iterator]
        for m in range(band_iterator):
            fusion_cupy = cp.array(fusioned_image[:, :, band_iterator])
            matrix_tmp_cupy = cp.array(matrix_tmp[:, :, m])
            num_cp = cp.vdot(fusion_cupy, matrix_tmp_cupy)
            den_cp = cp.vdot(matrix_tmp_cupy, matrix_tmp_cupy)
            num = num_cp.get()
            den = den_cp.get()
            result = num / den
            results_operations.append(result)
            matrix_tmp_gpu[:, :, band_iterator] = gpuarray.to_gpu(
                matrix_tmp[:, :, band_iterator].astype(np.float32)) - result * gpuarray.to_gpu(
                matrix_tmp[:, :, m].astype(np.float32))
            matrix_tmp = matrix_tmp_gpu.get()

    return matrix_tmp


def create_bands(tmp_matrix, pan_float):
    num_bands = int(tmp_matrix.shape[2])
    image_list = []
    band_iterator = 1
    while band_iterator < num_bands:
        image = tmp_matrix[:, :, band_iterator]
        float_image = image.astype(np.float32)
        image_list.append(float_image)
        band_iterator = band_iterator + 1

    image_list.insert(0, pan_float)
    matrix_temp = np.stack(image_list, axis=2)
    return matrix_temp


def merge_bands(matrix_tmp):
    global results_operations
    temporal_bands_gpu = gpuarray.to_gpu(matrix_tmp)
    tmp_bands = np.empty_like(matrix_tmp)
    num_bands = int(matrix_tmp.shape[2])
    band_iterator = 0
    for n in range(num_bands):
        tmp_bands[:, :, n] = matrix_tmp[:, :, n]
        for m in range(n):
            temporal_bands_gpu[:, :, n] = gpuarray.to_gpu(tmp_bands[:, :, n].astype(np.float32)) + resultados[
                band_iterator] * gpuarray.to_gpu(matrix_tmp[:, :, m].astype(np.float32))
            tmp_bands = temporal_bands_gpu.get()
            band_iterator = band_iterator + 1
    return tmp_bands


def merge_image(bandas_temp):
    final_list = []
    num_bands = int(bandas_temp.shape[2])
    for band_iterator in range(1, num_bands):
        final_image = bandas_temp[:, :, band_iterator]
        float_image = final_image.astype(np.float32)
        greater_fitted_values = fit_greater_values(float_image)
        float_image = greater_fitted_values.astype(np.float32)
        negative_fitted_values = fit_negative_values(float_image)
        float_image = negative_fitted_values.astype(np.float32)
        final_image = float_image.astype(np.uint8)
        final_list.append(final_image)

    gram_fusion = np.stack(final_list, axis=2)
    return gram_fusion


def fit_greater_values(matrix):
    matrix = matrix.astype(np.float32)
    matrix_gpu = gpuarray.to_gpu(matrix)
    matrix_gpu_new = gpuarray.empty_like(matrix_gpu)
    fit_positive = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] > 255){z[i] = 255.0;}else{z[i] = x[i];}",
        "adjust_value")
    fit_positive(matrix_gpu, matrix_gpu_new)
    return matrix_gpu_new.get()


def fit_negative_values(matrix):
    matrix_gpu = gpuarray.to_gpu(matrix)
    new_matrix_gpu = gpuarray.empty_like(matrix_gpu)
    fit_negative = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] < 0){z[i] = 0.0;}else{z[i] = x[i];}",
        "adjust_value")
    fit_negative(matrix_gpu, new_matrix_gpu)
    return new_matrix_gpu.get()
