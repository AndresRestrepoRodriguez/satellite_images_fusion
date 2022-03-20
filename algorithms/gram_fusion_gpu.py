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

resultados = []


def fusion_gram_gpu(im_multi, im_pan):
    pan_float = im_pan.astype(np.float32)
    fusion = pan_float
    n_bandas = int(im_multi.shape[2])
    i = 0
    lista_bandas = []

    while i < n_bandas:
        banda = im_multi[:, :, i]
        banda_float = banda.astype(np.float32)
        fusion = fusion + banda_float
        banda_float = banda_float.astype(np.uint8)
        lista_bandas.append(banda_float)
        i = i + 1

    fusion_prom = fusion / 4
    lista_bandas.insert(0, fusion_prom)
    fusioned_image = np.stack((lista_bandas), axis=2)
    mat_escalar = calcular_escalar(fusioned_image)
    matriz_bandas = crear_bandas(mat_escalar, pan_float)
    bandas_fusion = fusion_bandas(matriz_bandas)
    imagen_final = fusion_imagen(bandas_fusion)
    return imagen_final


def calcular_escalar(fusioned_image):
    global resultados
    fusioned_float = fusioned_image.astype(np.float32)
    matriz_temp_gpu = gpuarray.to_gpu(fusioned_float)
    N = int(fusioned_image.shape[2])
    matriz_temp = np.empty_like(fusioned_image)

    for n in range(N):
        matriz_temp[:, :, n] = fusioned_image[:, :, n]

        for m in range(n):
            fusioned_cp = cp.array(fusioned_image[:, :, n])
            matriz_temp_cp = cp.array(matriz_temp[:, :, m])
            num_cp = cp.vdot(fusioned_cp, matriz_temp_cp)
            den_cp = cp.vdot(matriz_temp_cp, matriz_temp_cp)
            num = num_cp.get()
            den = den_cp.get()
            resultado = num / den
            resultados.append(resultado)
            matriz_temp_gpu[:, :, n] = gpuarray.to_gpu(
                matriz_temp[:, :, n].astype(np.float32)) - resultado * gpuarray.to_gpu(
                matriz_temp[:, :, m].astype(np.float32))
            matriz_temp = matriz_temp_gpu.get()

    return matriz_temp


def crear_bandas(matriz_temp, pan_float):
    z = int(matriz_temp.shape[2])
    lista_imagen = []
    j = 1
    while j < z:
        imagen = matriz_temp[:, :, j]
        imagen_float = imagen.astype(np.float32)
        lista_imagen.append(imagen_float)
        j = j + 1

    lista_imagen.insert(0, pan_float)
    matriz_temp = np.stack((lista_imagen), axis=2)
    return matriz_temp


def fusion_bandas(matriz_temp):
    global resultados
    bandas_temp_gpu = gpuarray.to_gpu(matriz_temp)
    bandas_temp = np.empty_like(matriz_temp)
    limit = int(matriz_temp.shape[2])
    k = 0
    for n in range(limit):
        bandas_temp[:, :, n] = matriz_temp[:, :, n]
        for m in range(n):
            bandas_temp_gpu[:, :, n] = gpuarray.to_gpu(bandas_temp[:, :, n].astype(np.float32)) + resultados[
                k] * gpuarray.to_gpu(matriz_temp[:, :, m].astype(np.float32))
            bandas_temp = bandas_temp_gpu.get()
            k = k + 1
    return bandas_temp


def fusion_imagen(bandas_temp):
    l = 1
    lista_finales = []
    limit2 = int(bandas_temp.shape[2])
    for l in range(1, limit2):
        imagen_final = bandas_temp[:, :, l]
        imagen_float = imagen_final.astype(np.float32)
        ajuste_mayores = ajustar_valores_mayores(imagen_float)
        imagen_float = ajuste_mayores.astype(np.float32)
        ajuste_negativos = ajustar_valores_negativos(imagen_float)
        imagen_float = ajuste_negativos.astype(np.float32)

        imagen_final = imagen_float.astype(np.uint8)
        lista_finales.append(imagen_final)

    f_gram = np.stack((lista_finales), axis=2)
    return f_gram


def ajustar_valores_mayores(matrix):
    matrix = matrix.astype(np.float32)
    matrix_gpu = gpuarray.to_gpu(matrix)
    matrix_gpu_new = gpuarray.empty_like(matrix_gpu)
    adjustment_values = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] > 255){z[i] = 255.0;}else{z[i] = x[i];}",
        "adjust_value")
    adjustment_values(matrix_gpu, matrix_gpu_new)

    return matrix_gpu_new.get()


def ajustar_valores_negativos(matrix):
    matrix_gpu = gpuarray.to_gpu(matrix)
    new_matrix_gpu = gpuarray.empty_like(matrix_gpu)
    ajustar_menores = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] < 0){z[i] = 0.0;}else{z[i] = x[i];}",
        "adjust_value")
    ajustar_menores(matrix_gpu, new_matrix_gpu)
    return new_matrix_gpu.get()
