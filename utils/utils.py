import osgeo.gdalnumeric as gdn
import numpy as np
import skimage.io
from osgeo import ogr, osr, gdal


def read_image(path_image):
    data_image = skimage.io.imread(path_image, plugin='tifffile')
    info_image = gdal.Open(path_image)
    return data_image, info_image


def save_image_with_info(path_image, image_to_inject, geographical_data):
    num_bands = image_to_inject.shape[-1]
    # info = gdal.Info(geographical_data)
    cols = geographical_data.RasterXSize
    rows = geographical_data.RasterYSize
    origen = geographical_data.GetGeoTransform()
    origen_x = origen[0]
    origen_y = origen[3]
    pixel_width = origen[1]
    pixel_height = origen[5]
    driver = gdal.GetDriverByName("GTiff")
    proy = geographical_data.GetProjection()
    out_raster = driver.Create(path_image, cols, rows, num_bands, gdal.GDT_Byte)
    out_raster.SetGeoTransform((origen_x, pixel_width, origen[2], origen_y, origen[4], pixel_height))
    if num_bands == 1:
        out_raster.GetRasterBand(1).WriteArray(image_to_inject)
        out_raster.GetRasterBand(1).SetNoDataValue(np.nan)
    else:
        for i in range(num_bands):
            band_index = i + 1
            band_to_write = image_to_inject[:, :, i]
            out_raster.GetRasterBand(band_index).WriteArray(band_to_write)
            out_raster.GetRasterBand(band_index).SetNoDataValue(np.nan)
    s_ref = osr.SpatialReference()
    s_ref.ImportFromWkt(proy)
    out_raster.SetProjection(s_ref.ExportToWkt())
    out_raster.FlushCache()


def save_image_without_info(path_image, image_to_save):
    skimage.io.imsave(path_image, image_to_save, plugin='tifffile')
