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


def fusionar_gpu(banda, filtro1, filtro2):
    filtro1_gpu = gpuarray.to_gpu(filtro1)
    filtro2_gpu = gpuarray.to_gpu(filtro2)
    banda_gpu = gpuarray.to_gpu(banda)
    division = misc.divide(banda_gpu, filtro2_gpu)
    multiplicacion = linalg.multiply(division, filtro1_gpu)
    fusion_gpu = banda_gpu + multiplicacion

    return fusion_gpu.get()


def ajustar_valores_negativos(matrix):
    matrix_gpu = gpuarray.to_gpu(matrix)
    matrix_gpu_new = gpuarray.empty_like(matrix_gpu)

    adjustment_values = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] < 0){z[i] = 0.0;}else{z[i] = x[i];}",
        "adjust_value")
    adjustment_values(matrix_gpu, matrix_gpu_new)

    return matrix_gpu_new.get()


def ajustar_valores_mayores(matrix):
    matrix_gpu = gpuarray.to_gpu(matrix)
    matrix_gpu_new = gpuarray.empty_like(matrix_gpu)
    adjustment_values = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] > 255){z[i] = 255.0;}else{z[i] = x[i];}",
        "adjust_value")
    adjustment_values(matrix_gpu, matrix_gpu_new)

    return matrix_gpu_new.get()


filtro1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * (1 / 9)
filtro2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * (1 / 9)


def fusion_paso_alto_gpu(multi, pan):
    linalg.init()
    listaunion = []
    n_bandas = int(multi.shape[2])
    double_pan = pan.astype(np.float32)

    filtro1_cp = cp.array(filtro1)
    filtro2_cp = cp.array(filtro2)
    pan_cp = cp.array(double_pan)

    imagen1_cp = filters.correlate(pan_cp, filtro1_cp, mode='constant')
    imagen1_cpu = imagen1_cp.get()

    imagen1_gpu = gpuarray.to_gpu(imagen1_cpu)

    imagen2_cp = filters.correlate(pan_cp, filtro2_cp, mode='constant')
    imagen2_cpu = imagen2_cp.get()
    imagen2_gpu = gpuarray.to_gpu(imagen2_cpu)

    imagen1_validada = ajustar_valores_negativos(imagen1_gpu.astype(np.float32))
    imagen2_validada = ajustar_valores_negativos(imagen2_gpu.astype(np.float32))

    i = 0
    while i < n_bandas:
        banda = multi[:, :, i]
        banda_float = banda.astype(np.float32)
        fusionbandas = fusionar_gpu(banda_float, imagen1_validada, imagen2_validada)

        fusionbandas = ajustar_valores_mayores(fusionbandas)
        to_image = fusionbandas.astype(np.uint8)
        listaunion.append(to_image)
        i = i + 1
    fusioned_image = np.stack((listaunion), axis=2)

    return fusioned_image