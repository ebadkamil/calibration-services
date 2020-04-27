"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from concurrent.futures import ThreadPoolExecutor
import os.path as osp
import os
import re
from scipy import constants
import time

import numpy as np
import xarray as xr

from karabo_data import DataCollection, by_index

from .assembler import ImageAssembler
from .descriptors import MovingAverage, PyFaiAzimuthalIntegrator
from ..helpers import find_proposal, timeit


class AzimuthalIntegration(object):

    constant = 1e-3 * constants.c * constants.h / constants.e

    _azimuthal_integrator = PyFaiAzimuthalIntegrator()
    _azimuthal_integrator_ma = MovingAverage()

    def __init__(self, proposal, run, dettype, ai_config, window=1):

        assert dettype in ["AGIPD", "LPD"]

        self.run_path = find_proposal(proposal, run)
        self.dettype = dettype
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

        self._image_assembler = ImageAssembler.for_detector(self.dettype)

    def integrate(self, train_index=None, dark_data={}):
        del self._azimuthal_integrator
        del self._azimuthal_integrator_ma

        pattern = f"(.+){self.dettype}(.+)"

        files = [os.path.join(self.run_path, f)
                 for f in os.listdir(self.run_path)
                 if f.endswith('.h5') and re.match(pattern, f)]

        if not files:
            return

        run = DataCollection.from_paths(files).select(
            [("*/DET/*", "image.data")]).select_trains(by_index[100:200])

        for tid, data in run.trains():
            # assemble images
            assembled = self._image_assembler.assemble_image(
                data, dark_data=dark_data)
            # integrate
            self._azimuthal_integrator = assembled

            momentum, intensities = self._azimuthal_integrator
