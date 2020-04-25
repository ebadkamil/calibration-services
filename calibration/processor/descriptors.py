"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator


class IterativeHistogram(object):
    """Iterative Histogram descriptor.

    Stores and evaluate histogram of data iteratively.
    """
    def __init__(self, pixel_hist=False):
        """Attribute
        _pixel_hist: bool (True: if pixel wise histogram to be evaluated)
        """
        self._pixel_hist = pixel_hist
        self._histogram = None
        self._bin_edges = None

    def __get__(self, instance, cls):
        if instance is None:
            return self
        return (self._bin_edges, self._histogram)

    def __delete__(self, instance):
        self._bin_edges = None
        self._histogram = None

    def __set__(self, instance, data):
        if data is None:
            return

        if not isinstance(data, tuple):
            raise AttributeError(
                "Attribute must be a tuple. (bin_edges, image)")

        bin_edges, image = data
        self._bin_edges = bin_edges

        if len(image.shape) not in [2, 3]:
            raise AttributeError(
                f"Image must be 2 or 3 dimensional. Got {image.shape}")

        if len(image.shape) == 2:
            # Treat all data as 3 dimensional. First dim be memory cells
            image = image[np.newaxis, ...]

        if not self._pixel_hist:
            """Evaluate histogram over entire module"""
            counts_pr = []
            def _eval_stat(pulse):
                counts, _ = np.histogram(
                    image[pulse, ...].ravel(), bins=bin_edges)
                return counts

            with ThreadPoolExecutor(max_workers=5) as executor:
                for ret in executor.map(_eval_stat, range(image.shape[0])):
                    counts_pr.append(ret)

            if self._histogram is not None:
                self._histogram += np.stack(counts_pr)
            else:
                self._histogram = np.stack(counts_pr)

        else:
            """Evaluate histogram over each pixel"""
            def multihist(chunk, data, bin_edges, ret):
                start, end = chunk
                temp = data[:, start:end, :]
                bin_ix = np.searchsorted(bin_edges[1:], temp)

                X, Y, Z = temp.shape
                xgrid, ygrid, zgrid = np.meshgrid(
                    np.arange(X),
                    np.arange(Y),
                    np.arange(Z),
                    indexing='ij')

                counts = np.zeros((X, Y, Z, len(bin_edges)), dtype=np.uint32)

                np.add.at(counts, (xgrid, ygrid, zgrid, bin_ix), 1)
                ret[:, start:end, :, :] = counts[..., :-1]

            counts = np.zeros(
                image.shape + (len(bin_edges)-1,), dtype=np.uint32)

            start = 0
            chunk_size = 32
            chunks = []

            while start < counts.shape[1]:
                chunks.append(
                    (start, min(start + chunk_size, counts.shape[1])))
                start += chunk_size

            with ThreadPoolExecutor(max_workers=16) as executor:
                for chunk in chunks:
                    executor.submit(
                        multihist, chunk, image, bin_edges, counts)

            if self._histogram is not None:
                self._histogram += counts
            else:
                self._histogram = counts

    @property
    def pixel_hist(self):
        return self._pixel_hist

    @pixel_hist.setter
    def pixel_hist(self, val):
        self._pixel_hist = val


class MovingAverage(object):
    """Moving average data descriptor"""
    def __set_name__(self, owner, name):
        self.name = name

    def __init__(self, window=1):
        self._window = window
        self._ma_data = 0
        self._data_queue = deque()

    def __get__(self, instance, cls):
        if instance is None:
            return self
        try:
            data = instance.__dict__.get(
                self.name) / (len(self._data_queue) or 1)
            return data
        except TypeError:
            return

    def __set__(self, instance, data):
        if data is None:
            return

        self._data_queue.append(data)
        self._ma_data += data
        if len(self._data_queue) > self._window:
            self._ma_data -= self._data_queue.popleft()
        instance.__dict__[self.name] = self._ma_data

    def __delete__(self, instance):
        self._ma_data = 0
        self._data_queue.clear()

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, window):
        self._window = window


class PyFaiAzimuthalIntegrator(object):

    def __init__(self):
        self._distance = None
        self._wavelength = None
        self._poni1 = None
        self._poni2 = None
        self._intg_method = None
        self._intg_rng = None
        self._intg_pts = None
        self._pixel_size = None

        self._momentum = None
        self._intensities = None

        self._ai_integrator = None

    def __get__(self, instance, cls):
        if instance is None:
            return self

        return self._momentum, self._intensities

    def __set__(self, instance, data):
        # data is of shape (pulses, px, py)
        integrator = self._update_integrator()
        itgt1d = partial(integrator.integrate1d,
                         method=self._intg_method,
                         radial_range=self._intg_rng,
                         correctSolidAngle=True,
                         polarization_factor=1,
                         unit="q_A^-1")

        integ_points = self._intg_pts

        def _integrate(i):
            ret = itgt1d(data[i], integ_points)
            return ret.radial, ret.intensity

        with ThreadPoolExecutor(max_workers=5) as executor:
            rets = executor.map(_integrate,
                                range(data.shape[0]))

        momentums, intensities = zip(*rets)
        self._momentum = momentums[0]
        self._intensities = intensities

    def __delete__(self, instance):
        self._ai_integrator = None
        self._momentum = None
        self._intensities = None

    def _update_integrator(self):
        if self._ai_integrator is None:
            self._ai_integrator = AzimuthalIntegrator(
                dist=self._distance,
                pixel1=self._pixel_size,
                pixel2=self._pixel_size,
                poni1=self._poni1,
                poni2=self._poni2,
                rot1=0,
                rot2=0,
                rot3=0,
                wavelength=self._wavelength)
        else:
            if self._ai_integrator.dist != self._distance \
                    or self._ai_integrator.wavelength != self._wavelength \
                    or self._ai_integrator.poni1 != self._poni1 \
                    or self._ai_integrator.poni2 != self._poni2:
                self._ai_integrator.set_param(
                    (self._distance,
                     self._poni1,
                     self._poni2,
                     0,
                     0,
                     0,
                     self._wavelength))
        return self._ai_integrator

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, val):
        self._distance = val

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, val):
        self._wavelength = val

    @property
    def poni1(self):
        return self._poni1

    @poni1.setter
    def poni1(self, val):
        self._poni1 = val

    @property
    def poni2(self):
        return self._poni2

    @poni2.setter
    def poni2(self, val):
        self._poni2 = val

    @property
    def intg_method(self):
        return self._intg_method

    @intg_method.setter
    def intg_method(self, val):
        self._intg_method = val

    @property
    def intg_rng(self):
        return self._intg_rng

    @intg_rng.setter
    def intg_rng(self, val):
        self._intg_rng = val

    @property
    def intg_pts(self):
        return self._intg_pts

    @intg_pts.setter
    def intg_pts(self, val):
        self._intg_pts = val

    @property
    def pixel_size(self):
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, val):
        self._pixel_size = val
