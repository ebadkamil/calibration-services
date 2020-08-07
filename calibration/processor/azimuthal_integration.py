"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp
import os
import re
from scipy import constants

import numpy as np
import xarray as xr

from .assembler import ImageAssembler
from .descriptors import MovingAverage, PyFaiAzimuthalIntegrator
from ..helpers import (
    detector_data_collection, timeit)


class AzimuthalIntegration(object):
    """
    Parameters:
    -----------
    proposal: str, int
        A proposal number, such as 2012, '2012', 'p002012', or a path such as
        '/gpfs/exfel/exp/SPB/201701/p002012'.
    run: str, int
        A run number such as 243, '243' or 'r0243'.
    dettype: (str) AGIPD, LPD
    ai_config: dict
        For eg.:ai_config = dict(energy=9.3,
                                 pixel_size=0.5e-3,
                                 centrex=580,
                                 centrey=620,
                                 distance=0.2,
                                 intg_rng=[0.2, 5],
                                 intg_method='BBox',
                                 intg_pts=512)
    data: (str) "raw" or "proc"
        Default is "proc"
    window: (int) Moving average window size

    momentum: xarray
        Labelled xarray dims = ("trainId, "int_pts")
        Shape of numpy array: (n_trains, n_points)
    intensities: xarray
        Labelled xarray dims = ("trainId, "mem_cells",  "int_pts")
        Shape of numpy array: (n_trains, n_pulses, n_points)
    intensities_ma: np.ndarray of shape (n_pulses, n_points)"""

    constant = 1e-3 * constants.c * constants.h / constants.e

    _azimuthal_integrator = PyFaiAzimuthalIntegrator()
    _azimuthal_integrator_ma = MovingAverage()

    def __init__(self, proposal, run, dettype, ai_config, data='proc', window=1):
        dettype = dettype.upper()
        assert dettype in ["AGIPD", "LPD"]

        # Set properties of _azimuthal_integrator descriptor
        self.__class__._azimuthal_integrator.distance = ai_config['distance']
        self.__class__._azimuthal_integrator.wavelength = \
            AzimuthalIntegration.constant / ai_config["energy"]
        self.__class__._azimuthal_integrator.poni1 = \
            ai_config["centrey"] * ai_config['pixel_size']
        self.__class__._azimuthal_integrator.poni2 = \
            ai_config["centrex"] * ai_config['pixel_size']
        self.__class__._azimuthal_integrator.intg_method = \
            ai_config['intg_method']
        self.__class__._azimuthal_integrator.intg_rng = \
            ai_config['intg_rng']
        self.__class__._azimuthal_integrator.intg_pts = \
            ai_config['intg_pts']
        self.__class__._azimuthal_integrator.pixel_size = \
            ai_config['pixel_size']

        # Set window size for moving avg _azimuthal_integrator_ma descriptor
        self.__class__._azimuthal_integrator_ma.window = window

        self._image_assembler = ImageAssembler.for_detector(dettype)

        self.run = detector_data_collection(
            proposal, run, dettype, data=data)

        self.momentums = None
        self.intensities = None
        self.intensities_ma = None

    @timeit("Azimuthal Integration")
    def integrate(self, pulse_ids=None, dark_data={}, train_index=None):
        """
        pulse_ids: str
            For eg. ":" to select all pulses in a train
                    "start:stop:step" to select indices with certain step size
                    "1,2,3" comma separated pulse index to select specific pulses
                    "1,2,3, 5:10" mix of above two
            Default: all pulses ":"
        dark_data: dict optional
            dark_run[module_number] of shape (n_pulses, slow_scan, fast_scan)
            Default: empty dict {},
            If provided dark data will be subtracted from images
        train_index: by_index[start:stop] object from extra_data
            Default: None.
            If provided then only start:stop trains will be calculated
        """
        del self._azimuthal_integrator
        del self._azimuthal_integrator_ma

        if train_index is not None:
            self.run = self.run.select_trains(train_index)

        momentums = []
        intensities = []
        train_ids = []

        for tid, data in self.run.trains():
            # assemble images
            assembled = self._image_assembler.assemble_image(
                data,
                pulse_ids=pulse_ids,
                dark_data=dark_data,
                use_out_arr=True)

            if assembled is None:
                continue
            # set descriptor to integrate
            self._azimuthal_integrator = assembled

            # get momentum and intensity
            momentum, intensity = self._azimuthal_integrator
            intensity = np.stack(intensity)
            # set descriptor to calculate moving average
            self._azimuthal_integrator_ma = intensity

            momentums.append(momentum)
            intensities.append(intensity)
            train_ids.append(tid)

        if intensities:
            coords = {'trainId': np.array(train_ids)}
            dims = ['trainId', 'mem_cells', 'int_pts']

            self.momentums = xr.DataArray(
                data=np.stack(momentums), dims=['trainId', 'int_pts'],
                coords=coords)
            self.intensities = xr.DataArray(
                data=np.stack(intensities), dims=dims, coords=coords)

            self.intensities_ma = self._azimuthal_integrator_ma

            return self.momentums, self.intensities, self.intensities_ma


class ImageIntegrator:
    """
    Parameters:
    -----------
    ai_config: dict
        For eg.:ai_config = dict(energy=9.3,
                                 pixel_size=0.5e-3,
                                 centrex=580,
                                 centrey=620,
                                 distance=0.2,
                                 intg_rng=[0.2, 5],
                                 intg_method='BBox',
                                 intg_pts=512,
                                 threshold_mask=(0,100),
                                 user_mask=user_mask)
        user_mask: ndarray (Same shape as image to integrate)

    momentum: ndarray
        Shape of numpy array: (n_points, )
    intensities: ndarray
        Shape of numpy array: (n_pulses, n_points)"""
    constant = 1e-3 * constants.c * constants.h / constants.e

    _azimuthal_integrator = PyFaiAzimuthalIntegrator()

    def __init__(self, ai_config):

        # Set properties of _azimuthal_integrator descriptor
        self.__class__._azimuthal_integrator.distance = ai_config['distance']
        self.__class__._azimuthal_integrator.wavelength = \
            AzimuthalIntegration.constant / ai_config["energy"]
        self.__class__._azimuthal_integrator.poni1 = \
            ai_config["centrey"] * ai_config['pixel_size']
        self.__class__._azimuthal_integrator.poni2 = \
            ai_config["centrex"] * ai_config['pixel_size']
        self.__class__._azimuthal_integrator.intg_method = \
            ai_config['intg_method']
        self.__class__._azimuthal_integrator.intg_rng = \
            ai_config['intg_rng']
        self.__class__._azimuthal_integrator.intg_pts = \
            ai_config['intg_pts']
        self.__class__._azimuthal_integrator.pixel_size = \
            ai_config['pixel_size']
        self.__class__._azimuthal_integrator.threshold_mask = \
            ai_config.get('threshold_mask', None)
        self.__class__._azimuthal_integrator.user_mask = \
            ai_config.get('user_mask', None)

        self.momentums = None
        self.intensities = None

    def integrate(self, image):
        """
        image: ndarray
            Shape: (pulses, px, py)
        """
        self._azimuthal_integrator = image
        self.momentums, self.intensities = self._azimuthal_integrator
        return self.momentums, np.array(self.intensities)