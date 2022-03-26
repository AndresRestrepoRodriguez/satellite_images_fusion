import gdal
import osgeo.gdalnumeric as gdn
import numpy as np
import skimage.io


def read_image(path_image):
  data_image = skimage.io.imread(path_image, plugin='tifffile')
  info_image = gdal.Open(path_image)
  return data_image, info_image


def save_image_with_info(path_image, image_to_inject, geographical_data):
  num_bands = image_to_inject.shape[-1]
  info = gdal.Info(geographical_data)
  cols = geographical_data.RasterXSize
  rows = geographical_data.RasterYSize
  origen = geographical_data.GetGeoTransform()
  origenX = origen[0]
  origenY = origen[3]
  pixelWidth=origen[1]
  pixelHeight=origen[5]
  driver = gdal.GetDriverByName("GTiff")
  proy = geographical_data.GetProjection()
  outRaster = driver.Create(path_image, cols, rows, num_bands, gdal.GDT_Byte)
  outRaster.SetGeoTransform((origenX, pixelWidth, origen[2], origenY,origen[4], pixelHeight))
  if num_bands ==1:
      outRaster.GetRasterBand(1).WriteArray(image_to_inject)
      outRaster.GetRasterBand(1).SetNoDataValue(np.nan)
  else:
      for i in range(num_bands):
          band_index = i+1
          band_to_write = image_to_inject[:,:,i]
          outRaster.GetRasterBand(band_index).WriteArray(band_to_write)
          outRaster.GetRasterBand(band_index).SetNoDataValue(np.nan)

  s_ref = osr.SpatialReference()
  s_ref.ImportFromWkt(proy)
  outRaster.SetProjection(s_ref.ExportToWkt())
  outRaster.FlushCache()


def save_image_without_info(path_image, image_to_inject):
    skimage.io.imsave(path_image, image_to_inject, plugin='tifffile')

