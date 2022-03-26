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


def fusion_hpf_gpu(im_pan, im_multi):
    lista_float = []
    i = 0
    double_pan = im_pan.astype(np.float32)
    n_bandas = int(im_multi.shape[2])

    while i < n_bandas:
        banda = im_multi[:, :, i]
        bandafloat = banda.astype(np.float32)
        lista_float.append(bandafloat)
        i = i + 1
    img = crear_filtro(double_pan)
    var = calcular_varianza(im_multi, n_bandas)
    final_img = fusionar_imagen(im_pan, var, lista_float, n_bandas, img)

    return final_img


def crear_filtro(double_pan):
    filtro = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, 80, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1]]) * (1 / 106);

    pan_cp = cp.array(double_pan)
    filtro_cp = cp.array(filtro)
    imagen_cp = filters.correlate(pan_cp, filtro_cp, mode='constant')
    imagen_cpu = imagen_cp.get()
    double_imagen_cpu = imagen_cpu.astype(np.float32)
    imagen1 = ajustar_valores_negativos(imagen_cpu)

    return imagen1


def ajustar_valores_negativos(matrix):
    matrix_gpu = gpuarray.to_gpu(matrix)
    new_matrix_gpu = gpuarray.empty_like(matrix_gpu)
    ajustar_menores = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] < 0){z[i] = 0.0;}else{z[i] = x[i];}",
        "adjust_value")
    ajustar_menores(matrix_gpu, new_matrix_gpu)
    return new_matrix_gpu.get()


def calcular_varianza(im_multi, n_bandas):
    ceros = np.zeros((n_bandas, 2))
    for n in range(n_bandas):
        ceros[n][0] = n
        matriz_gpu = gpuarray.to_gpu(im_multi[:, :, n].astype(np.float32))
        varianza_gpu = misc.std(matriz_gpu)
        ceros[n][1] = varianza_gpu.get()
    return ceros


def fusionar_imagen(im_pan, ceros, lista_float, n_bandas, imagen1):
    pan_gpu = gpuarray.to_gpu(im_pan.astype(np.float32))
    std_pan_gpu = misc.std(pan_gpu)
    sum_var = 0
    for i in ceros:
        sum_var += i[1]
    total_var = sum_var / (std_pan_gpu * 0.65)
    lista_bases = []
    imagen_gpu = gpuarray.to_gpu(imagen1)
    for j in range(n_bandas):
        lista_gpu = gpuarray.to_gpu(lista_float[j])
        mult = imagen_gpu * total_var.get()
        base_temp = lista_gpu + mult
        matriz_new = ajustar_valores_mayores(base_temp)
        base = matriz_new.astype(np.uint8)
        lista_bases.append(base)

    fusioned_image = np.stack((lista_bases), axis=2)
    return fusioned_image


def ajustar_valores_mayores(matrix):
    matrix_gpu = matrix.astype(np.float32)
    matrix_gpu_new = gpuarray.empty_like(matrix_gpu)
    adjustment_values = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] > 255){z[i] = 255.0;}else{z[i] = x[i];}",
        "adjust_value")
    adjustment_values(matrix_gpu, matrix_gpu_new)

    return matrix_gpu_new.get()
