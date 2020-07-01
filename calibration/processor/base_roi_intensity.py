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
import plotly.graph_objects as go
import xarray as xr

from .descriptors import MovingAverage

from ..gui.plots import ScatterPlot
from ..helpers import (
    control_data_collection, detector_data_collection,
    parse_ids, timeit)


class BaseRoiIntensity(object):
    """Base class to evaluate ROI intensities

    Attributes:
    -----------
    modno: str, int
        Channel number between 0, 15
    proposal: str, int
        A proposal number, such as 2012, '2012', 'p002012', or a path such as
        '/gpfs/exfel/exp/SPB/201701/p002012'.
    run: str, int
        A run number such as 243, '243' or 'r0243'.
    dettype: (str) AGIPD, LPD
    window: (int) Moving average window size
    roi_intensity: xarray
        Labelled xarray dims = ("trainId, rois, mem_cells")
        Shape of numpy array: (n_trains, n_rois, n_pulses)
    roi_intensity_ma: xarray
        Moving averaged roi_intensity over trains"""
    _intensity_ma = MovingAverage()

    def __init__(self, modno, proposal, run, dettype, window=1):

        if not isinstance(modno, int):
            modno = int(modno)

        assert modno in range(16)
        dettype = dettype.upper()
        assert dettype in ["AGIPD", "LPD"]

        self.run = detector_data_collection(
            proposal, run, dettype, modno=modno)
        self.control = control_data_collection(
            proposal, run)

        self.dettype = dettype
        self.modno = modno

        self.__class__._intensity_ma.window = window
        self.rois = None
        self.roi_intensity = None
        self.roi_intensity_ma = None

    def __call__(self, **kwargs):
        """kwargs should be a subset of kwargs in
           method :eval_module_roi_intensity:"""
        roi_intensity_ma = self.eval_module_roi_intensity(**kwargs)
        return roi_intensity_ma

    @timeit("Module ROI intensity")
    def eval_module_roi_intensity(
        self, rois=None, pulse_ids=None,
        dark_run=None, gain=None,
        use_normalizer=None):
        """
        pulse_ids: str
            For eg. ":" to select all pulses in a train
                    "start:stop:step" to select indices with certain step size
                    "1,2,3" comma separated pulse index to select specific pulses
                    "1,2,3, 5:10" mix of above two
            Default: all pulses ":"
        rois: list
            In case of one roi: [x0, x1, y0, y1]
            For multiple rois: [[x0, x1, y0, y1], [x0, x1, y0, y1], ...]
        dettype: str
            "AGIPD", "LPD"
        dark_run: (numpy.ndarray) or dict optional
            dark_data shape (n_pulses, slow_scan, fast_scan)
            dark_run[module_number] of shape (n_pulses, slow_scan, fast_scan)
            Default: None,
            If provided dark data will be subtracted from images
        use_normalizer: tuple
            (source_name, property)
        """

        # Reset moving average just in case if called twice for same instance
        del self._intensity_ma

        module = [key for key in self.run.instrument_sources
                  if re.match(r"(.+)/DET/(.+):(.+)", key)]

        if len(module) != 1:
            return

        self.run = self.run.select([(module[0], "image.data")])

        pulse_ids = ":" if pulse_ids is None else pulse_ids
        self.pulses = parse_ids(pulse_ids)
        self.rois = rois

        intensities = []
        train_ids = []
        intensities_ma = []
        for tid, data in self.run.trains():
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

            intensity = np.stack(
                [np.mean(img, axis=(-1, -2)) for img in self.roi_images])

            # set _intensity_ma
            self._intensity_ma = intensity

            intensities.append(intensity)
            #get _intensity_ma and append to intensitie_ma
            intensities_ma.append(self._intensity_ma)
            train_ids.append(tid)

        if intensities:
            coords = {'trainId': np.array(train_ids)}
            dims = ['trainId', 'rois', 'mem_cells']
            data = xr.DataArray(
                data=np.stack(intensities), dims=dims, coords=coords)

            self.roi_intensity = data
            self.roi_intensity_ma = xr.DataArray(
                data=np.stack(intensities_ma), dims=dims, coords=coords)

            if use_normalizer is not None:
                self.normalize(use_normalizer)
            return self.roi_intensity_ma

    def plot_scan(self, src, prop):
        """Plot roi_intensity wrt to scan variable.
           Scan variable should be one value per train id.
        src: str
            karabo device ID
        prop: str
            karabo property
        Return:
        -------
        fig: plotly Figure object
            use fig.show() to render in notebooks
        """
        if self.roi_intensity is None:
            print("Roi intensity not available")
            return

        if self.control is None:
            print("Control data collection object is not available")
            return

        scan_data = self.control.get_array(src, prop)

        assert len(scan_data.shape) == 1

        align = xr.merge(
            [self.roi_intensity.rename('roi_intensity'),
             scan_data.rename('scan_data')],
             join='inner')

        # Take mean and std after grouping with scan data
        mean_align = align.groupby('scan_data').mean(dim=['trainId'])
        std_align = align.groupby('scan_data').std(dim=['trainId'])

        # Create ScatterPlot object
        fig = ScatterPlot(title=f'Module {self.modno}',
                          xlabel=f"Scan variable ({src}/{prop})",
                          ylabel="Mean ROI intensity",
                          legend='Pulse index',
                          drop_down_label="ROI")

        # Set data
        fig.setData(
            mean_align['scan_data'],
            mean_align['roi_intensity'],
            yerror=std_align['roi_intensity'])
        return fig

    def correct(self, offset=None, gain=None):
        """Hook to use for correction of images
           Implement in inherited class"""
        pass

    def normalize(self):
        """Hook to use for Normalizion of roi_intensity
           Implement in inherited class"""
        pass
