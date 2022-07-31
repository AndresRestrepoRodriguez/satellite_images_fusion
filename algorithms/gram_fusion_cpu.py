import numpy as np

results_operations = []


def fusion_gram_cpu(multispectral_image, panchromatic_image):
    panchromatic_float = panchromatic_image.astype(np.float32)
    panchromatic_copy = panchromatic_float
    num_bands = int(multispectral_image.shape[2])
    band = 0
    bands_list = []

    while band < num_bands:
        local_band = multispectral_image[:, :, band]
        float_band = local_band.astype(np.float32)
        panchromatic_copy = panchromatic_copy + float_band
        float_band = float_band.astype(np.uint8)
        bands_list.append(float_band)
        band = band + 1

    total_bands = num_bands + 1
    fusion_mean = panchromatic_copy / total_bands
    bands_list.insert(0, fusion_mean)
    fusioned_image = np.stack(bands_list, axis=2)
    mat_scalar = get_scalar(fusioned_image)
    bands_matrix = create_bands(mat_scalar, panchromatic_float)
    fusion_bands = merge_bands(bands_matrix)
    final_image = merge_image(fusion_bands)
    return final_image


def get_scalar(fusioned_image):
    global results_operations
    num_bands = int(fusioned_image.shape[2])
    matriz_temp = np.empty_like(fusioned_image)

    for band in range(num_bands):
        matriz_temp[:, :, band] = fusioned_image[:, :, band]
        for m in range(band):
            num = np.vdot(fusioned_image[:, :, band], matriz_temp[:, :, m])
            den = np.vdot(matriz_temp[:, :, m], matriz_temp[:, :, m])
            result_tmp = num / den
            results_operations.append(result_tmp)
            matriz_temp[:, :, band] = matriz_temp[:, :, band] - result_tmp * matriz_temp[:, :, m]
    return matriz_temp


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
    tmp_matrix = np.stack(image_list, axis=2)
    return tmp_matrix


def merge_bands(matrix_tmp):
    global results_operations
    temporal_bands = np.empty_like(matrix_tmp)
    num_bands = int(matrix_tmp.shape[2])
    band_iterator = 0
    for band_value in range(num_bands):
        temporal_bands[:, :, band_value] = matrix_tmp[:, :, band_value]
        for m in range(band_value):
            temporal_bands[:, :, band_value] = temporal_bands[:, :, band_value] + results_operations[band_iterator] * matrix_tmp[:, :, m]
            band_iterator = band_iterator + 1
    return temporal_bands


def merge_image(bandas_temp):
    final_list = []
    num_bands = int(bandas_temp.shape[2])
    for band_iterator in range(1, num_bands):
        final_image = bandas_temp[:, :, band_iterator]
        final_image[final_image > 255] = 255
        final_image[final_image < 0] = 0
        final_image = final_image.astype(np.uint8)
        final_list.append(final_image)

    gram_fusion = np.stack(final_list, axis=2)
    return gram_fusion
