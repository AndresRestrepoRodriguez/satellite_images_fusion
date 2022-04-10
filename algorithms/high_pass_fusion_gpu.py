import pycuda.autoinit
import numpy as np
import skimage.io
from scipy import ndimage
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
import skcuda.misc as misc
import skimage
from pycuda.elementwise import ElementwiseKernel
from cupyx.scipy.ndimage import filters
import cupy as cp

initial_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * (1 / 9)
second_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * (1 / 9)


def fusion_gpu(band, initial_filter, second_filter):
    initial_filter_gpu = gpuarray.to_gpu(initial_filter)
    second_filter_gpu = gpuarray.to_gpu(second_filter)
    band_gpu = gpuarray.to_gpu(band)
    division = misc.divide(band_gpu, second_filter_gpu)
    multiplication = linalg.multiply(division, initial_filter_gpu)
    fusioned_gpu = band_gpu + multiplication
    return fusioned_gpu.get()


def fit_negative_values(matrix):
    matrix_gpu = gpuarray.to_gpu(matrix)
    matrix_gpu_new = gpuarray.empty_like(matrix_gpu)
    fit_negative = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] < 0){z[i] = 0.0;}else{z[i] = x[i];}",
        "adjust_value")
    fit_negative(matrix_gpu, matrix_gpu_new)
    return matrix_gpu_new.get()


def fit_greater_values(matrix):
    matrix_gpu = gpuarray.to_gpu(matrix)
    matrix_gpu_new = gpuarray.empty_like(matrix_gpu)
    fit_positive = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] > 255){z[i] = 255.0;}else{z[i] = x[i];}",
        "adjust_value")
    fit_positive(matrix_gpu, matrix_gpu_new)
    return matrix_gpu_new.get()


def fusion_high_pass_gpu(multi, pan):
    linalg.init()
    union_list = []
    num_bands = int(multi.shape[2])
    panchromatic_float = pan.astype(np.float32)
    initial_filter_cupy = cp.array(initial_filter)
    second_filter_cupy = cp.array(second_filter)
    panchromatic_cupy = cp.array(panchromatic_float)

    image_initial_filter_cupy = filters.correlate(panchromatic_cupy, initial_filter_cupy, mode='constant')
    image_initial_filter_cpu = image_initial_filter_cupy.get()
    image_initial_filter_gpu = gpuarray.to_gpu(image_initial_filter_cpu)

    image_second_filter_cupy = filters.correlate(panchromatic_cupy, second_filter_cupy, mode='constant')
    image_second_filter_cpu = image_second_filter_cupy.get()
    image_second_filter_gpu = gpuarray.to_gpu(image_second_filter_cpu)

    fitted_negative_initial_filter = fit_negative_values(image_initial_filter_gpu.astype(np.float32))
    fitted_negative_second_filter = fit_negative_values(image_second_filter_gpu.astype(np.float32))

    band_iterator = 0
    while band_iterator < num_bands:
        band = multi[:, :, band_iterator]
        float_band = band.astype(np.float32)
        fusion_bands = fusion_gpu(float_band, fitted_negative_initial_filter, fitted_negative_second_filter)

        fusion_bands = fit_greater_values(fusion_bands)
        result_image = fusion_bands.astype(np.uint8)
        union_list.append(result_image)
        band_iterator = band_iterator + 1
    fusioned_image = np.stack(union_list, axis=2)
    return fusioned_image
