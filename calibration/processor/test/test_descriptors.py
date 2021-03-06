"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
import unittest

from calibration.processor.descriptors import (
    IterativeHistogram, MovingAverage, PyFaiAzimuthalIntegrator)


class TestIterativeHistogram(unittest.TestCase):
    _iterative_hist = IterativeHistogram()

    def setUp(self):
        self.bin_edges = np.linspace(-200, 400, 601)
        self.data = np.random.uniform(-200, 400, (10, 20, 30))

    def test_attributes(self):
        with self.assertRaisesRegex(
            AttributeError, "Attribute must be a tuple"):
            self._iterative_hist = self.data

        with self.assertRaisesRegex(
            AttributeError, "Image must be 2 or 3 dimensional"):
            self._iterative_hist = (
                self.bin_edges, np.random.uniform(200, 400, (10,)))

    def test_histogram(self):
        self._iterative_hist = self.bin_edges, self.data

        _, hist = self._iterative_hist

        self.assertEqual(hist.shape, (10, 600))
        np.testing.assert_array_equal(hist[0], np.histogram(
            self.data[0], bins=600, range=(-200, 400))[0])

    def test_pixel_histogram(self):
        del self._iterative_hist
        self.assertIsNone(self.__class__._iterative_hist._histogram)
        self.assertIsNone(self.__class__._iterative_hist._bin_edges)

        self.__class__._iterative_hist.pixel_hist = True
        self._iterative_hist = self.bin_edges, self.data

        _, hist = self._iterative_hist

        self.assertEqual(
            hist.shape, self.data.shape + (len(self.bin_edges)-1,))


class TestMovingAverage(unittest.TestCase):
    _moving_average = MovingAverage(window=2)

    def test_moving_average(self):
        data = np.full((4, 6), 1.0)
        self._moving_average = data

        np.testing.assert_array_equal(self._moving_average, data)

        data = np.full((4, 6), 2.0)
        self._moving_average = data
        np.testing.assert_array_equal(
            self._moving_average, np.full((4, 6), 1.5))

        data = np.full((4, 6), 3.0)
        self._moving_average = data
        np.testing.assert_array_equal(
            self._moving_average, np.full((4, 6), 2.5))

        # Test window size change
        del self._moving_average
        self.__class__._moving_average.window = 3

        self.assertEqual(self.__class__._moving_average._ma_data, 0)
        self.assertEqual(len(self.__class__._moving_average._data_queue), 0)

        data = np.full((4, 6), 4.0)
        self._moving_average = data
        np.testing.assert_array_equal(
            self._moving_average, np.full((4, 6), 4.0))
