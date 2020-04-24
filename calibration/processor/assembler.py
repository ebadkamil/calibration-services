"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import re

from karabo_data import RunDirectory, stack_detector_data
from karabo_data.geometry2 import AGIPD_1MGeometry, LPD_1MGeometry

from ..helpers import timeit


class ImageAssembler(object):
    """ImageAssembler Class"""

    class BaseAssembler(object):
        def __init__(self, geom_file=None, quad_prositions=None):
            self.geom_file = geom_file
            self.quad_prositions = quad_prositions
            self.geom = None

        def _get_modules_data(train_data, dark_data=None):
            raise NotImplementedError

        def get_geom_object(self):
            pass

        def assemble_image(self, train_data, dark_data=None):
            modules_data = self._get_modules_data(
                train_data, dark_data=dark_data)
            if modules_data is None:
                return

            self.get_geom_object()

            if self.geom is not None:
                assembled, center = self.geom.position_all_modules(modules_data)
            else:
                assembled = modules_data

            return assembled


    @classmethod
    def for_detector(cls, detector, geom_file=None, quad_prositions=None):
        detector = detector.upper()

        if detector == "LPD":
            return cls.LpdAssembler(
                geom_file=geom_file, quad_prositions=quad_prositions)

        elif detector == "AGIPD":
            return cls.AgipdAssembler(
                geom_file=geom_file, quad_prositions=quad_prositions)

        else:
            raise NotImplementedError("Detector assembler not implemented")
