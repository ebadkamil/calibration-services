"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import copy
import numpy as np
import unittest

from extra_geom import AGIPD_1MGeometry

from calibration.processor.assembler import ImageAssembler

class TestAgipdAssembler(unittest.TestCase):

    def setUp(self):
        self._assembler = ImageAssembler.for_detector("AGIPD")
        key = "image.data"
        self._raw_train_data = {
            'SPB_DET_AGIPD1M-1/DET/11CH0:xtdf':
                {key: np.ones((20, 2, 512, 128), dtype=np.uint16)},
            'SPB_DET_AGIPD1M-1/DET/7CH0:xtdf':
                {key: np.ones((20, 2, 512, 128), dtype=np.uint16)},
            'SPB_DET_AGIPD1M-1/DET/8CH0:xtdf':
                {key: np.ones((20, 2, 512, 128), dtype=np.uint16)},
            'SPB_DET_AGIPD1M-1/DET/3CH0:xtdf':
                {key: np.ones((20, 2, 512, 128), dtype=np.uint16)},
        }

        self._proc_train_data = {
            'SPB_DET_AGIPD1M-1/DET/11CH0:xtdf':
                {key: np.ones((20, 512, 128), dtype=np.float32)},
            'SPB_DET_AGIPD1M-1/DET/7CH0:xtdf':
                {key: np.ones((20, 512, 128), dtype=np.float32)},
            'SPB_DET_AGIPD1M-1/DET/8CH0:xtdf':
                {key: np.ones((20, 512, 128), dtype=np.float32)},
            'SPB_DET_AGIPD1M-1/DET/3CH0:xtdf':
                {key: np.ones((20, 512, 128), dtype=np.float32)},
        }

    def test_geom_object(self):
        self.assertIsNone(self._assembler.geom)
        self._assembler.get_geom_object()
        self.assertIsNotNone(self._assembler.geom)
        self.assertTrue(isinstance(self._assembler.geom, AGIPD_1MGeometry))

    def test_get_modules_data(self):
        train_data = copy.deepcopy(self._raw_train_data)
        stacked_data = self._assembler._get_modules_data(
            train_data, [-1])

        self.assertEqual(stacked_data.dtype, np.float32)
        self.assertEqual(stacked_data.shape, (20, 16, 512, 128))

        # Test for 4 pulses
        train_data = copy.deepcopy(self._raw_train_data)
        stacked_data = self._assembler._get_modules_data(
            train_data, [0, 1, 2, 5])

        self.assertEqual(stacked_data.dtype, np.float32)
        self.assertEqual(stacked_data.shape, (4, 16, 512, 128))

        # Test for modules with different pulses in one train
        train_data = copy.deepcopy(self._raw_train_data)
        train_data['SPB_DET_AGIPD1M-1/DET/11CH0:xtdf']["image.data"] = \
            np.ones((2, 2, 512, 128), dtype=np.uint16)

        stacked_data = self._assembler._get_modules_data(
                train_data, [-1])
        self.assertIsNone(stacked_data)

        # Test for calibrated data
        train_data = copy.deepcopy(self._proc_train_data)
        stacked_data = self._assembler._get_modules_data(
                train_data, [-1])
        self.assertEqual(stacked_data.shape, (20, 16, 512, 128))

    def test_assemble_image(self):
        train_data = copy.deepcopy(self._raw_train_data)
        assembled = self._assembler.assemble_image(train_data)
        self.assertEqual(assembled.dtype, np.float32)
        self.assertEqual(assembled.shape[0], 20)
        self.assertTrue(assembled.shape[1] > 1024)

        # Test out array was assigned
        train_data = copy.deepcopy(self._raw_train_data)
        self.assertIsNone(self._assembler.out_array)
        assembled = self._assembler.assemble_image(
            train_data, use_out_arr=True)
        self.assertIsNotNone(self._assembler.out_array)


class TestJungFrauAssembler(unittest.TestCase):

    def setUp(self):
        self._assembler = ImageAssembler.for_detector("JungFrau")
        key = "data.adc"
        self._train_data_pre_20 = {
            'FXE_XAD_JF1M/DET/RECEIVER-1:daqOutput':
                {key: np.ones((1, 512, 1024), dtype=np.uint16)},
            'FXE_XAD_JF1M/DET/RECEIVER-2:daqOutput':
                {key: np.ones((1, 512, 1024), dtype=np.uint16)},
        }

        self._train_data_post_20 = {
            'FXE_XAD_JF1M/DET/JNGFR01:daqOutput':
                {key: np.ones((1, 512, 1024), dtype=np.uint16)},
            'FXE_XAD_JF1M/DET/JNGFR02:daqOutput':
                {key: np.ones((1, 512, 1024), dtype=np.uint16)},
        }

    def test_assemble_image(self):
        train_data = copy.deepcopy(self._train_data_pre_20)
        assembled = self._assembler.assemble_image(train_data)
        self.assertEqual(assembled.dtype, np.float32)
        self.assertEqual(assembled.shape[0], 1)
        self.assertEqual(assembled.shape, (1, 2, 512, 1024))

        train_data = copy.deepcopy(self._train_data_post_20)
        assembled = self._assembler.assemble_image(train_data)
        self.assertEqual(assembled.dtype, np.float32)
        self.assertEqual(assembled.shape[0], 1)
        self.assertEqual(assembled.shape, (1, 2, 512, 1024))