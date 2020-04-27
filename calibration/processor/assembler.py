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

from ..helpers import timeit, parse_ids


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
            self.out_array = None

        def _get_modules_data(train_data, pulses, dark_data={}):
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

        def assemble_image(self, train_data,
            pulse_ids=None, dark_data={}, use_out_arr=False):

            pulse_ids = ":" if pulse_ids is None else pulse_ids
            pulses = parse_ids(pulse_ids)

            modules_data = self._get_modules_data(
                train_data, pulses, dark_data=dark_data)
            if modules_data is None:
                return

            self.get_geom_object()

            if self.geom is not None:

                if use_out_arr and self.out_array is None:
                    image_dtype = modules_data.dtype
                    n_images = (modules_data.shape[0], )

                    self.out_array = self.geom.output_array_for_position_fast(
                        extra_shape=n_images, dtype=image_dtype)

                assembled, _ = self.geom.position_all_modules(
                    modules_data, out=self.out_array)
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

        def _get_modules_data(self, train_data, pulses, dark_data={}):

            def _corrections(source):
                pattern = "(.+)/DET/(.+)CH0:xtdf"
                modno = int((re.match(pattern, source)).group(2).strip())
                try:
                    image = train_data[source]["image.data"]
                except KeyError:
                    return

                if pulses != [-1] and image.shape[0] != 0:
                    image = image[pulses, ...]

                if image.dtype == np.uint16:
                    # Raw image
                    image = image.squeeze(axis=1)
                    image = image.astype(np.float32)

                if dark_data and image.shape[0] != 0:
                    if pulses != [-1]:
                        dark = dark_data[str(modno)][pulses, ...]
                    else:
                        dark = dark_data[str(modno)][0:image.shape[0], ...]
                    image -= dark

                train_data[source]["image.data"] = image

            with ThreadPoolExecutor(
                    max_workers=len(train_data.keys())) as executor:
                for source in train_data.keys():
                    executor.submit(_corrections, source)
            # stack detector data
            try:
                stacked_data = stack_detector_data(train_data, "image.data")
            except (ValueError, IndexError, KeyError) as e:
                print(e)
                return

            if stacked_data.shape[0] != 0:
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

        def _get_modules_data(self, train_data, pulses, dark_data=None):
            def _corrections(source):
                pattern = "(.+)/DET/(.+)CH0:xtdf"
                modno = int((re.match(pattern, source)).group(2).strip())
                try:
                    image = train_data[source]["image.data"]
                except KeyError:
                    return

                if pulses != [-1] and image.shape[0] != 0:
                    image = image[pulses, ...]

                if image.dtype == np.uint16:
                    # Raw image
                    image = image.squeeze(axis=1)
                    image = image.astype(np.float32)

                if dark_data and image.shape[0] != 0:
                    if pulses != [-1]:
                        dark = dark_data[str(modno)][pulses, ...]
                    else:
                        dark = dark_data[str(modno)][0:image.shape[0], ...]
                    image -= dark

                train_data[source]["image.data"] = image

            with ThreadPoolExecutor(
                    max_workers=len(train_data.keys())) as executor:
                for source in train_data.keys():
                    executor.submit(_corrections, source)

            # stack detector data
            try:
                stacked_data = stack_detector_data(train_data, "image.data")
            except (ValueError, IndexError, KeyError) as e:
                print(e)
                return

            if stacked_data.shape[0] != 0:
                return stacked_data
