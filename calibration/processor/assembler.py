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

    @classmethod
    def for_detector(cls, detector, geom_file=None, quad_prositions=None):
        """
        detector: (str) for eg. LPD, AGIPD
        geom_file: (str) path to geometry file
        quad_positions: (list) quadrant positions"""
        detector = detector.upper()

        if detector == "LPD":
            return cls.LpdAssembler(
                geom_file=geom_file, quad_prositions=quad_prositions)

        elif detector == "AGIPD":
            return cls.AgipdAssembler(
                geom_file=geom_file, quad_prositions=quad_prositions)

        else:
            raise NotImplementedError("Detector assembler not implemented")

    class BaseAssembler(object):
        """BaseAssembler class"""
        def __init__(self, geom_file=None, quad_prositions=None):
            """
            Attributes
            ----------
            _geom_file: (str) path to geometry file
            _quad_positions: (list) list of quadrant positions
            """
            self._geom_file = geom_file
            self._quad_prositions = quad_prositions
            self.geom = None

        def _get_modules_data(train_data, dark_data=None):
            """stack modules data together from train_dictionary

            train_data: A nested dictionary returned from extra_data
                        :method: train_from_id, train_from_index etc
            dark_data: dict
                dark_data[module_number] is an ndarray
                if provided: then this data should be subtracted from
                    image data for the corresponding module

            return:
                stacked data: (pulses, modules, px, py)
            """
            raise NotImplementedError

        def get_geom_object(self):
            """Create extra-geom geometry object"""
            pass

        def assemble_image(self, train_data, dark_data=None):
            modules_data = self._get_modules_data(
                train_data, dark_data=dark_data)
            if modules_data is None:
                return

            self.get_geom_object()

            if self.geom is not None:
                assembled, _ = self.geom.position_all_modules(modules_data)
            else:
                assembled = modules_data

            return assembled

    class AgipdAssembler(BaseAssembler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def get_geom_object(self):
            if self._geom_file is not None:
                try:
                    self.geom = AGIPD_1MGeometry.from_crystfel_geom(
                        self._geom_file)
                except Exception as ex:
                    print(ex)
            else:
                self.geom = AGIPD_1MGeometry.from_quad_positions(
                    quad_pos=[
                        (-525, 625),
                        (-550, -10),
                        (520, -160),
                        (542.5, 475), ])

        def _get_modules_data(self, train_data, dark_data=None):

            def _corrections(source, train_data=train_data):
                pattern = "(.+)/DET/(.+)CH0:xtdf"
                modno = int((re.match(pattern, source)).group(2).strip())

                image = train_data[source]["image.data"][:, 0, ...]

                image = image.astype(np.float32)

                if dark_data is not None and image.shape[0] != 0:
                    image -= dark_data[str(modno)][0:image.shape[0], ...]

                train_data[source]["image.data"] = image

            with ThreadPoolExecutor(
                    max_workers=len(train_data.keys())) as executor:
                for source in train_data.keys():
                    executor.submit(_corrections, source)
            # assemble image
            try:
                stacked_data = stack_detector_data(train_data, "image.data")
            except (ValueError, IndexError, KeyError) as e:
                print(e)
                return
            return stacked_data

    class LpdAssembler(BaseAssembler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def get_geom_object(self):
            if all([self._geom_file is not None,
                    self._quad_prositions is not None]):
                try:
                    self.geom = LPD_1MGeometry.from_h5_file_and_quad_positions(
                        self._geom_file, self._quad_prositions)
                except Exception as ex:
                    print(ex)
            else:
                self.geom = LPD_1MGeometry.from_quad_positions(
                    quad_pos=[
                        [11.4, 299],
                        [-11.5, 8],
                        [254.5, -16],
                        [278.5, 275]],)

        def _get_modules_data(self, train_data, dark_data=None):

            def _corrections(source, train_data=train_data):
                pattern = "(.+)/DET/(.+)CH0:xtdf"
                modno = int((re.match(pattern, source)).group(2).strip())

                image = np.squeeze(
                    train_data[source]["image.data"], axis=1)

                image = image.astype(np.float32)

                if dark_data is not None and image.shape[0] != 0:
                    image -= dark_data[str(modno)][0:image.shape[0], ...]

                train_data[source]["image.data"] = image

            with ThreadPoolExecutor(
                    max_workers=len(train_data.keys())) as executor:
                for source in train_data.keys():
                    executor.submit(_corrections, source)
            # assemble image
            try:
                stacked_data = stack_detector_data(train_data, "image.data")
            except (ValueError, IndexError, KeyError) as e:
                print(e)
                return
            return stacked_data
