import gdal
import osgeo.gdalnumeric as gdn
import numpy as np


def cargar_info(input_file, dim_ordering="channels_last", dtype='float32'):
    file = gdal.Open(input_file, gdal.GA_ReadOnly)
    bands = [file.GetRasterBand(i) for i in range(1, file.RasterCount + 1)]
    bandas = np.array([gdn.BandReadAsArray(band) for band in bands]).astype(dtype)
    if dim_ordering == "channels_last":
        bandas = np.transpose(bandas, [1, 2, 0])  # Reorders dimensions, so that channels are last

    info = gdal.Info(file)
    return bandas, info
