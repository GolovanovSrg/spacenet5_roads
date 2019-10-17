import math

import albumentations as alb
import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from utils import normalize_image, save_zipped_pickle, load_zipped_pickle, create_mask


def convert_array_to_multichannel(in_arr, n_channels=7, append_total_band=False):
    h, w = in_arr.shape[:2]
    out_arr = np.zeros((h, w, n_channels), dtype='uint8')
    
    for band in range(n_channels):
        val = band + 1
        band_out = np.zeros((h, w), dtype='uint8')
        band_arr_bool = np.where(in_arr == val)
        band_out[band_arr_bool] = 1
        out_arr[..., band] = band_out
 
    if append_total_band:
        tot_band = np.zeros((h, w), dtype='uint8')
        band_arr_bool = np.where(in_arr > 0)
        tot_band[band_arr_bool] = 1
        tot_band = tot_band.reshape(h, w, 1)
        out_arr = np.concatenate((out_arr, tot_band), axis=-1)
    
    return out_arr



class TrainTransform:
    def __init__(self, crops_size):
        self.transform = alb.Compose([alb.RandomCrop(*crops_size),
                                      alb.HorizontalFlip(p=0.5),
                                      alb.VerticalFlip(p=0.5),
                                      alb.Transpose(0.5),
                                      alb.ShiftScaleRotate(shift_limit=0.0625,
                                                           scale_limit=0.10,
                                                           rotate_limit=30,
                                                           p=0.75)])

    def __call__(self, image, mask):
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']


class TestTransform:
    def __init__(self, image_size):
        self.transform = alb.Compose([alb.PadIfNeeded(*image_size)])

    def __call__(self, image, mask):
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']


class Spacenet5Dataset(Dataset):
    def __init__(self, data, channels=None, transform=None, cache_dir=None):
        super().__init__()

        self.data = data
        self.channels = channels
        self.transform = transform
        self.cache_dir = cache_dir

        if cache_dir is not None:
            cache_dir.mkdir(exist_ok=True)

    def _get_image(self, image_path):
        image = imread(str(image_path), plugin='tifffile')
        if self.channels is not None:
            image = image[..., self.channels]

        return normalize_image(image)

    def _get_cached_mask_path(self, geojson_path):
        if self.cache_dir is not None:
            return self.cache_dir / geojson_path.name.replace('geojson', 'zpkl')
        return None

    def _get_mask(self, image_path, geojson_path):
        def speed_to_bin(speed_mph, bin_size_mph=10):
            return int(math.ceil(speed_mph / bin_size_mph))

        cached_mask_path = self._get_cached_mask_path(geojson_path)

        if cached_mask_path is not None and cached_mask_path.exists():
            return load_zipped_pickle(cached_mask_path)

        mask = create_mask(str(image_path), str(geojson_path), speed_to_bin,
                            buffer_distance_meters=2, buffer_roundness=1,
                            dissolve_by='inferred_speed_mph', zero_frac_thresh=0.05)
        mask = convert_array_to_multichannel(mask, n_channels=7, append_total_band=True)

        if cached_mask_path is not None:
            save_zipped_pickle(mask, cached_mask_path)

        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, geojosn_path = self.data[idx]

        image = self._get_image(image_path)
        mask = self._get_mask(image_path, geojosn_path)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).long()

        return image, mask, idx
