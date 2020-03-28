"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import fnmatch
import os.path as osp
import os
import re
import time

import numpy as np
import xarray as xr

from karabo_data import DataCollection, by_index, H5File

from ..helpers import pulse_filter, parse_ids, find_proposal, timeit


class BaseRoiIntensity(object):

    def __init__(self, modno, proposal, run, dettype):

        if not isinstance(modno, int):
            modno = int(modno)

        assert modno in range(16)
        assert dettype in ["AGIPD", "LPD"]
        self.modno = modno
        self.run_path = find_proposal(proposal, run)
        self.dettype = dettype

        self.rois = None
        self.roi_intensity = None

    def eval_module_roi_intensity(
        self, rois=None, pulse_ids=None,
        dark_run=None, gain=None,
        use_normalizer=None):

        pattern = f"(.+){self.dettype}{self.modno:02d}(.+)"

        files = [os.path.join(self.run_path, f) for f in os.listdir(self.run_path)
                 if f.endswith('.h5') and re.match(pattern, f)]

        if not files:
            return

        run = DataCollection.from_paths(files)

        module = [key for key in run.instrument_sources
                  if re.match(r"(.+)/DET/(.+):(.+)", key)]

        if len(module) != 1:
            return

        run = run.select([(module[0], "image.data")]) # for debug .select_trains(by_index[100:200])

        pulse_ids = ":" if pulse_ids is None else pulse_ids
        self.pulses = parse_ids(pulse_ids)
        self.rois = rois

        intensities = []
        train_ids = []
        for tid, data in run.trains():
            if self.dettype == 'LPD':
                image = np.squeeze(
                    data[module[0]]["image.data"], axis=1) # (pulses, 1, ss, fs)
            else:
                image = data[module[0]]["image.data"][:, 0, ...]

            if image.shape[0] == 0:
                continue

            self.roi_images = [image]

            if rois is not None:
                if not isinstance(rois[0], list):
                    rois = [rois]
                self.roi_images = [
                    image[..., x0:x1, y0:y1] for x0, x1, y0, y1 in rois]

            if self.pulses != [-1]:
                self.roi_images = [
                    img[self.pulses, ...].astype(np.float32)
                    for img in self.roi_images] 
            else:
                self.roi_images = [ 
                    img.astype(np.float32) for img in self.roi_images] 

            self.correct(offset=dark_run, gain=gain)

            intensities.append(
                np.stack([np.mean(img, axis=(-1, -2)) for img in self.roi_images]))
            train_ids.append(tid)

        if intensities:
            coords = {'trainId': np.array(train_ids)}
            dims = ['trainId', 'rois', 'mem_cells']
            data = xr.DataArray(
                data=np.stack(intensities), dims=dims, coords=coords)

            self.roi_intensity = data

            if use_normalizer is not None:
                self.normalize(use_normalizer)

    def correct(self, offset=None, gain=None):
        pass

    def normalize(self):
        pass


class AgipdRoiIntensity(BaseRoiIntensity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def correct(self, offset=None, gain=None):
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

