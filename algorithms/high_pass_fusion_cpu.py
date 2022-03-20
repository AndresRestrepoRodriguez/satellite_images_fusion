import numpy as np
import skimage
from scipy import ndimage

filtro1 = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]]) * (1/9)
filtro2 = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]]) * (1/9)

def fusion_paso_alto(banda,im1, im2):
    fusionbanda = banda + np.multiply(np.divide(banda, im2), im1)
    return fusionbanda

def fusion_paso_alto_cpu(im_multi, im_pan):
    listaunion = []
    n_bandas = int(im_multi.shape[2])
    double_pan = im_pan.astype(np.float32)
    imagen1 = ndimage.correlate(double_pan, filtro1, mode='constant')
    imagen2 = ndimage.correlate(double_pan, filtro2, mode='constant')

    imagen1[imagen1<0] = 0
    imagen2[imagen2<0] = 0


    double_imagen1 = imagen1.astype(np.float32)
    double_imagen2 = imagen2.astype(np.float32)
    i = 0

    while i < n_bandas:
        banda = im_multi[:,:,i]
        bandafloat = banda.astype(np.float32)
        fusionbandas = fusion_paso_alto(bandafloat, double_imagen1, double_imagen2)
        fusionbandas[fusionbandas>255] = 255
        to_image = fusionbandas.astype(np.uint8)
        listaunion.append(to_image)
        i = i + 1

    fusioned_image = np.stack((listaunion),axis = 2)
    return fusioned_image
