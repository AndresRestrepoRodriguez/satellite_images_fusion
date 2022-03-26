import numpy as np
import skimage


def operacion_bandas(color_ban, ban_pan):
    return (color_ban + ban_pan) * 0.5


def fusion_valor_medio_cpu(im_multi, im_pan):
    listaunion = []
    n_bandas = int(im_multi.shape[2])
    pan_float = im_pan.astype(np.float32)
    i = 0

    while i < n_bandas:
        matrix = im_multi[:, :, i]
        matrixfloat = matrix.astype(np.float32)
        fusionbandas = operacion_bandas(matrixfloat, pan_float)
        union = fusionbandas.astype(np.uint8)
        listaunion.append(union)
        i = i + 1

    fusioned_image = np.stack((listaunion), axis=2)

    return fusioned_image
