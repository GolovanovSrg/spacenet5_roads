import gzip
import pickle
import random
import tempfile
import os

import geopandas as gpd
import numpy as np
import torch

from osgeo import gdal, ogr, osr
from skimage.io import imread

from tools import osmnx_funcs


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


def train_val_split(data, val_size=0.2):
    val_data_size = int(len(data) * val_size)
    random.shuffle(data)
    val_data = data[:val_data_size]
    train_data = data[val_data_size:]

    return train_data, val_data


def get_spacenet5_data(data_dir, image_type='PS-MS'):
    assert image_type in ['MS', 'PAN', 'PS-MS', 'PS-RGB']

    def get_geojson_name(image_name):
        geojson_name = image_name.replace(image_type, 'geojson_roads_speed') \
                                 .replace('tif', 'geojson')
        return geojson_name

    images_dir = data_dir / image_type
    geojsons_dir = data_dir / 'geojson_roads_speed'

    images = list(sorted(images_dir.glob('*.tif')))

    if not geojsons_dir.exists():
        return images

    geojsons = [geojsons_dir / get_geojson_name(image.name) for image in images]

    return list(zip(images, geojsons))


def normalize_image(image):
    float_image = image.astype('float32') / 2047  # WorldView-3: 11 bit per channel
    mean = np.mean(float_image, axis=(0, 1))
    std = np.std(float_image, axis=(0, 1))

    return (float_image - mean) / std

def save_zipped_pickle(obj, filename, protocol=3):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


def gdf_to_array(gdf, im_file, output_raster, mask_burn_val_key='', burnValue=150, NoData_value=0):
    assert output_raster.endswith('.tif')

    # set target info
    gdata = gdal.Open(im_file)
    target_ds = gdal.GetDriverByName('GTiff').Create(output_raster,
                                                     gdata.RasterXSize,
                                                     gdata.RasterYSize, 1,
                                                     gdal.GDT_Byte,
                                                     ['COMPRESS=LZW'])
    target_ds.SetGeoTransform(gdata.GetGeoTransform())

    # set raster info
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(gdata.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)

    outdriver = ogr.GetDriverByName('MEMORY')
    outDataSource = outdriver.CreateDataSource('memData')
    tmp = outdriver.Open('memData', 1)
    outLayer = outDataSource.CreateLayer("states_extent", raster_srs, geom_type=ogr.wkbMultiPolygon)

    # burn
    burnField = "burn"
    idField = ogr.FieldDefn(burnField, ogr.OFTInteger)
    outLayer.CreateField(idField)
    featureDefn = outLayer.GetLayerDefn()
    for j, geomShape in enumerate(gdf['geometry'].values):
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(ogr.CreateGeometryFromWkt(geomShape.wkt))
        burnVal = int(gdf[mask_burn_val_key].values[j]) if len(mask_burn_val_key) > 0 else burnValue
        outFeature.SetField(burnField, burnVal)
        outLayer.CreateFeature(outFeature)

    if len(mask_burn_val_key) > 0:
        gdal.RasterizeLayer(target_ds, [1], outLayer, options=["ATTRIBUTE=%s" % burnField])
    else:
        gdal.RasterizeLayer(target_ds, [1], outLayer, burn_values=[burnVal])


def create_mask(image_path, geojson_path, bin_conversion_func,
                buffer_distance_meters=2, buffer_roundness=1,
                dissolve_by='inferred_speed_mph', zero_frac_thresh=0.05):

    """
    Create buffer around geojson for speeds, use bin_conversion_func to
    assign values to the mask
    """

    # get gdf_buffer
    try:
        inGDF = gpd.read_file(geojson_path)
    except:
        h, w = imread(image_path, plugin='tifffile').shape[:2]
        return np.zeros((h, w), dtype='uint8')

    if len(inGDF) == 0:
        h, w = imread(image_path, plugin='tifffile').shape[:2]  
        return np.zeros((h, w), dtype='uint8')
        
    # project
    projGDF = osmnx_funcs.project_gdf(inGDF)
    gdf_utm_buffer = projGDF.copy()

    # perform Buffer to produce polygons from Line Segments
    gdf_utm_buffer['geometry'] = gdf_utm_buffer.buffer(buffer_distance_meters,
                                                       buffer_roundness)
    gdf_utm_dissolve = gdf_utm_buffer.dissolve(by=dissolve_by)
    gdf_utm_dissolve.crs = gdf_utm_buffer.crs
    gdf_buffer = gdf_utm_dissolve.to_crs(inGDF.crs)

    # set burn values
    values = gdf_buffer.index.values
    burned_values = [bin_conversion_func(v) for v in values]
    gdf_buffer['burned_values'] = burned_values

    # create mask
    with tempfile.TemporaryDirectory() as tmpdirname: 
        mask_path = os.path.join(tmpdirname, 'mask.tif')
        gdf_to_array(gdf_buffer, image_path, mask_path, 'burned_values')
        mask = imread(mask_path, as_gray=True, plugin='tifffile')

    # check to ensure no mask outside the image pixels (some images are largely black)
    im_gray = imread(image_path, as_gray=True, plugin='tifffile')
    # check if im_gray is more than X percent black
    zero_frac = 1 - float(np.count_nonzero(im_gray)) / im_gray.size
    if zero_frac >= zero_frac_thresh:
        # ensure the label doesn't extend beyond the image
        zero_locs = np.where(im_gray == 0)
        # set mask_gray to zero at location of zero_locs
        mask[zero_locs] = 0
        
    return mask 
