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
    """ModuleRoiIntensity class to correct and normalize roi_intensity"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def correct(self, offset=None, gain=None):
        """Subtract offset from roi_images"""
        if offset is not None:
            if not isinstance(offset, np.ndarray): # passed as a dict
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


class GainAdjustedRoiIntensity(BaseRoiIntensity):
    """ModuleRoiIntensity class to correct and normalize roi_intensity"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def correct(self, offset=None, gain=None):
        """Subtract offset and apply gain to roi_images"""
        if offset is not None and gain is not None:
            high = offset["high"]
            medium = offset["medium"]
            low = offset["low"]

            gain_high = gain['high']
            gain_medium = gain['medium']
            gain_low = gain['low']

            if all([isinstance(high, dict), 
                    isinstance(medium, dict), 
                    isinstance(low, dict)]):
                high = high[self.modno]
                medium = medium[self.modno]
                low = low[self.modno]

            high_roi_images = [high]
            medium_roi_images = [medium]
            low_roi_images = [low]
            rois = self.rois
            if rois is not None:
                if not isinstance(rois[0], list):
                    rois = [rois]
                high_roi_images = [
                    high[..., x0:x1, y0:y1] for x0, x1, y0, y1 in rois]
                medium_roi_images = [
                    medium[..., x0:x1, y0:y1] for x0, x1, y0, y1 in rois]
                low_roi_images = [
                    low[..., x0:x1, y0:y1] for x0, x1, y0, y1 in rois]

            if self.pulses != [-1]:
                high_roi_images = [
                    img[self.pulses, ...] for img in high_roi_images]
                medium_roi_images = [
                    img[self.pulses, ...] for img in medium_roi_images]
                low_roi_images = [
                    img[self.pulses, ...] for img in low_roi_images]

            for i in range(len(self.roi_images)):
                image = self.roi_images[i]
                high = high_roi_images[i]
                medium = medium_roi_images[i]
                low = low_roi_images[i]
                
                currim = np.zeros_like(image)
            
                # High gain correction
                corrim = (image - high) * gain_high
                currim[np.where(image <= 4096)] = corrim[np.where(image <= 4096)]

                # Medium gain correction
                corrim = (image - medium) * gain_medium
                currim[np.where(np.logical_and(image <= 8192, image > 4096))] = \
                    corrim[np.where(np.logical_and(image <= 8192, image > 4096))]

                # Low gain correction
                corrim = (image - low) * gain_low
                currim[np.where(image > 8192)] = corrim[np.where(image > 8192)]

                self.roi_images[i] = currim
