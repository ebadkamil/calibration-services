"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np


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
    def __init__(self, window=1):
        self._window = window
        self._ma_data = None
        self._data_queue = deque()

    def __get__(self, instance, cls):
        if instance is None:
            return self
        return self._ma_data / (len(self._data_queue) or 1)

    def __set__(self, instance, data):
        if data is None:
            return

        if all([self._ma_data is not None, 
                self._window > 1,
                # data.shape == self._ma_data.shape
                ]):
            self._data_queue.append(data)
            self._ma_data += data
            if len(self._data_queue) > self._window:
                self._ma_data -= self._data_queue.popleft()

        else:
            self._data_queue.append(data)
            self._ma_data = data

    def __delete__(self, instance):
        self._ma_data = None
        self._data_queue.clear()

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, window):
        self._window = window

