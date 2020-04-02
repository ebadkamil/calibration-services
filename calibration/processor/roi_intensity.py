"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import fnmatch
import os

import numpy as np
import xarray as xr

from karabo_data import DataCollection
from .base_roi_intensity import BaseRoiIntensity


class ModuleRoiIntensity(BaseRoiIntensity):
    """AgipdRoiIntensity class to correct and normalize roi_intensity"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def correct(self, offset=None, gain=None):
        """Subtract offset from roi_images"""
        if offset is not None:
            if not isinstance(offset, np.ndarray): # passed as a dict
                try:
                    dark_data = offset[str(self.modno)]
                except KeyError:
                    dark_data = offset[self.modno]
            else:
                dark_data = offset

            dark_roi_images = [dark_data]
            rois = self.rois
            if rois is not None:
                if not isinstance(rois[0], list):
                    rois = [rois]
                dark_roi_images = [
                    dark_data[..., x0:x1, y0:y1] for x0, x1, y0, y1 in rois]

            if self.pulses != [-1]:
                dark_roi_images = [
                    img[self.pulses, ...] for img in dark_roi_images]

            if not all(map(
                lambda x, y: x.shape == y.shape, self.roi_images, dark_roi_images)):
                raise ValueError("Shapes of image and dark data don't match")

            self.roi_images = [ 
                self.roi_images[i] - dark_roi_images[i] 
                for i in range(len(self.roi_images))]

    def normalize(self, normalizer):
        """Normalize roi_intensity with pulse resolved normalizer"""
        src, prop = normalizer
        files = [f for f in os.listdir(self.run_path) if f.endswith('.h5')]
        files = [os.path.join(self.run_path, f)
                 for f in fnmatch.filter(files, '*DA*')]

        normalizer_data = DataCollection.from_paths(files).get_array(
            src, prop, extra_dims=['mem_cells'])

        if self.pulses != [-1]:
            normalizer_data = normalizer_data[:, self.pulses]
        else:
            normalizer_data = normalizer_data[:, 0:self.roi_intensity.shape[-1]]

        self.roi_intensity, normalizer_data = xr.align(
            self.roi_intensity, normalizer_data)
        
        self.roi_intensity /= normalizer_data