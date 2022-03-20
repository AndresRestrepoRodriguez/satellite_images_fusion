import numpy as np
from scipy import ndimage


# imagen1 = []
# ceros=[]
def fusion_hpf_cpu(im_pan, im_multi):
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

    imagen1 = ndimage.correlate(double_pan, filtro, mode='constant').astype(np.int8)
    imagen1[imagen1 < 0] = 0
    return imagen1


def calcular_varianza(im_multi, n_bandas):
    ceros = np.zeros((n_bandas, 2))
    for n in range(n_bandas):
        ceros[n][0] = n
        varianza = np.std(im_multi[:, :, n])
        ceros[n][1] = varianza
    return ceros


def fusionar_imagen(im_pan, ceros, lista_float, n_bandas, imagen1):
    stdpan = np.std(im_pan)
    sum_var = 0
    for i in ceros:
        sum_var += i[1]
    total_var = sum_var / (stdpan * 0.65)
    lista_bases = []
    for j in range(n_bandas):
        base = lista_float[j] + imagen1 * total_var
        base[base > 255] = 255
        base = base.astype(np.uint8)
        lista_bases.append(base)

    fusioned_image = np.stack((lista_bases), axis=2)
    return fusioned_image


