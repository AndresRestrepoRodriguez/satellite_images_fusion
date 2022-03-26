import pycuda.autoinit
import numpy as np
import skimage.io
from scipy import ndimage
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray


def operacion_bandas_gpu(color_ban, ban_pan):
    color_ban_gpu = gpuarray.to_gpu(color_ban)
    pan_ban_gpu = gpuarray.to_gpu(ban_pan)
    resultado_op_bandas = (color_ban_gpu + pan_ban_gpu) * 0.5

    return resultado_op_bandas.get()


def fusion_valor_medio_gpu(im_multi, im_pan):
    listaunion = []
    n_bandas = int(im_multi.shape[2])
    pan_float = im_pan.astype(np.float32)
    i = 0
    while i < n_bandas:
        matrix = im_multi[:, :, i]
        matrixfloat = matrix.astype(np.float32)
        fusionbandas = operacion_bandas_gpu(matrixfloat, pan_float)
        union = fusionbandas.astype(np.uint8)
        listaunion.append(union)
        i = i + 1
    fusioned_image = np.stack((listaunion), axis=2)
    return fusioned_image
