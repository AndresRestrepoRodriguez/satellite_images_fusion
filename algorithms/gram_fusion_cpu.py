import numpy as np

resultados = []


def fusion_gram_cpu(im_multi, im_pan):
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
    N = int(fusioned_image.shape[2])
    matriz_temp = np.empty_like(fusioned_image)

    for n in range(N):
        matriz_temp[:, :, n] = fusioned_image[:, :, n]
        for m in range(n):
            num = np.vdot(fusioned_image[:, :, n], matriz_temp[:, :, m])
            den = np.vdot(matriz_temp[:, :, m], matriz_temp[:, :, m])
            resultado = num / den
            resultados.append(resultado)
            matriz_temp[:, :, n] = matriz_temp[:, :, n] - resultado * matriz_temp[:, :, m]
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
    bandas_temp = np.empty_like(matriz_temp)
    limit = int(matriz_temp.shape[2])
    k = 0
    for n in range(limit):
        bandas_temp[:, :, n] = matriz_temp[:, :, n]
        for m in range(n):
            bandas_temp[:, :, n] = bandas_temp[:, :, n] + resultados[k] * matriz_temp[:, :, m]
            k = k + 1
    return bandas_temp


def fusion_imagen(bandas_temp):
    l = 1
    lista_finales = []
    limit2 = int(bandas_temp.shape[2])
    for l in range(1, limit2):
        imagen_final = bandas_temp[:, :, l]
        imagen_final[imagen_final > 255] = 255
        imagen_final[imagen_final < 0] = 0
        imagen_final = imagen_final.astype(np.uint8)
        lista_finales.append(imagen_final)

    f_gram = np.stack((lista_finales), axis=2)
    return f_gram
