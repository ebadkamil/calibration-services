"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import h5py
import multiprocessing as mp
import os.path as osp
import os
import re
import numpy as np
import sys
import time

from karabo_data import DataCollection, by_index

from mpi4py import MPI

from .MPI import distribute
from ..helpers import pulse_filter, parse_ids


class MPIEvalHistogram:
    """MPI Class to evaluate histogram

    Attributes:
    -----------
        path: (str) Path to Run folder
        pixel_hist: (bool) optional
            Default: False. For pixel wise histogram set it to True
        dark_run: (numpy.ndarray) optional
            dark_dta shape (n_pulses, slow_scan, fast_scan)
            Default: None,
            If provided dark data will be subtracted from images"""
    _tag_mod = 111
    _tag_histogram_data = 1111
    _tag_histogram_shape = 1112
    _tag_histogram_dtype = 1113

    _tag_image_data = 1121
    _tag_image_shape = 1122
    _tag_image_dtype = 1123

    def __init__(self, path,
                 pixel_hist=False, comm=MPI.COMM_WORLD, dark_run=None):
        """Initialization"""
        self.comm = comm
        self.path = path
        self.pixel_hist = pixel_hist
        self.dark_run = dark_run

        self.modno = None
        self.histogram = None
        self.mean_image = None

    def process(self, bin_edges, pulse_ids=None):
        """Evaluate Histogram and mean image
        Parameters:
        -----------
            bin_edges: (np.1darray) required
            pulse_ids: str, optional
                Default: all pulses ":"
                For eg. ":" to select all pulses in a train
                "start:stop:step" to select indices with certain step size
                "1,2,3" comma separated pulse index to select specific pulses
                "1,2,3, 5:10" mix of above two
        """
        self.bin_edges = bin_edges
        pulse_ids = ":" if pulse_ids is None else pulse_ids
        self.pulses = parse_ids(pulse_ids)

        self.channels, self.local_sequences = distribute(
            self.path, comm=self.comm)

        self._eval_histogram()
        self._gather()

    def _gather(self):
        """Gather the results in master node
        Results are available in self.result attribute.
        TODO: Better implementation Using MPI Gather and scatter
        """
        if self.comm.rank == 0:
            self.result_hist = {mod: 0 for mod in self.channels}
            self.result_mean_image = {mod: [] for mod in self.channels}

            if self.modno is not None:
                self.result_hist[self.modno] += self.histogram
                self.result_mean_image[self.modno].append(self.mean_image)

            for source in range(1, self.comm.size):
                mod = self.comm.recv(source=source, tag=self._tag_mod)
                if mod is not None:
                    # Receive histograms from other workers
                    hist_dtype = self.comm.recv(
                        source=source, tag=self._tag_histogram_dtype)
                    hist_shape = self.comm.recv(
                        source=source, tag=self._tag_histogram_shape)

                    histogram = np.empty(hist_shape, dtype=hist_dtype)
                    self.comm.Recv(
                        histogram, source=source, tag=self._tag_histogram_data)

                    print("Recv Hist", hist_shape)
                    # Receive mean image from other workers
                    mean_image_dtype = self.comm.recv(
                        source=source, tag=self._tag_image_dtype)
                    mean_image_shape = self.comm.recv(
                        source=source, tag=self._tag_image_shape)
                    mean_image = np.empty(
                        mean_image_shape, dtype=mean_image_dtype)
                    self.comm.Recv(
                        mean_image, source=source, tag=self._tag_image_data)

                else:
                    histogram = self.comm.recv(source=source)
                    mean_image = self.comm.recv(source=source)

                if all(
                    [mod is not None,
                     histogram is not None,
                     mean_image is not None]):
                    self.result_hist[mod] += histogram
                    self.result_mean_image[mod].append(mean_image)

            for mod in self.result_mean_image:
                temp = np.mean(np.stack(self.result_mean_image[mod]), axis=0)
                self.result_mean_image[mod] = temp
                print("Image shape ", temp.shape)

        else:
            self.comm.send(self.modno, dest=0, tag=self._tag_mod)

            if self.modno is not None:
                hist_shape = self.histogram.shape
                hist_dtype = self.histogram.dtype

                self.comm.send(
                    hist_shape, dest=0, tag=self._tag_histogram_shape)
                self.comm.send(
                    hist_dtype, dest=0, tag=self._tag_histogram_dtype)
                self.comm.Send(
                    self.histogram, dest=0, tag=self._tag_histogram_data)

                mean_image_shape = self.mean_image.shape
                mean_image_dtype = self.mean_image.dtype
                print("Send image", mean_image_shape)
                self.comm.send(
                    mean_image_shape, dest=0, tag=self._tag_image_shape)
                self.comm.send(
                    mean_image_dtype, dest=0, tag=self._tag_image_dtype)
                self.comm.Send(
                    self.mean_image, dest=0, tag=self._tag_image_data)

            else:
                self.comm.send(self.histogram, dest=0)
                self.comm.send(self.mean_image, dest=0)

    def _eval_histogram(self):

        if not self.local_sequences:
            return

        modno, files = zip(*self.local_sequences)

        run = DataCollection.from_paths(
            list(files)).select_trains(by_index[:10])

        module = [key for key in run.instrument_sources
                  if re.match(r"(.+)/DET/(.+):(.+)", key)]

        if len(module) != 1:
            raise ValueError("More than one module found")

        num_trains = 0
        histogram = 0
        mean_image = 0
        for tid, data in run.trains(devices=[(module[0], "image.data")],
                                    require_all=True):

            image = data[module[0]]["image.data"][:, 0, ...]

            if image.shape[0] == 0:
                continue

            if self.pulses != [-1]:
                image = image[self.pulses, ...].astype(np.float32)
            else:
                image = image.astype(np.float32)

            if self.dark_run is not None:
                dark_data = self.dark_run[:, 0, ...]
                if image.shape == dark_data.shape:
                    image -= dark_data
                else:
                    raise ValueError(
                        f"Different data shapes, dark_data: {dark_data.shape}"
                        f" Run data: {image.shape}")

            mean_image += image
            if not self.pixel_hist:
                """Evaluate histogram over entire module"""
                counts_pr = []

                def _eval_stat(pulse):
                    counts, _ = np.histogram(
                        image[pulse, ...].ravel(), bins=self.bin_edges)
                    return counts

                with ThreadPoolExecutor(max_workers=5) as executor:
                    for ret in executor.map(_eval_stat, range(image.shape[0])):
                        counts_pr.append(ret)
                histogram += np.stack(counts_pr)

            else:
                """Evaluate histogram over each pixel"""
                def multihist(data, bin_edges):
                    bin_ix = np.searchsorted(bin_edges[1:], data)

                    X, Y, Z = data.shape
                    xgrid, ygrid, zgrid = np.meshgrid(
                        np.arange(X),
                        np.arange(Y),
                        np.arange(Z),
                        indexing='ij')

                    counts = np.zeros(
                        (X, Y, Z, len(bin_edges)), dtype=np.uint32)

                    np.add.at(counts, (xgrid, ygrid, zgrid, bin_ix), 1)
                    return counts[..., :-1]

                counts = multihist(image, self.bin_edges)
                histogram += counts

            num_trains += 1

        if num_trains != 0:
            self.modno = modno[0]
            self.histogram = histogram
            self.mean_image = mean_image / num_trains

    def hist_to_file(self, path):
        if path and self.comm.rank == 0:
            bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2.0

            with h5py.File(path, "w") as f:
                for modno, data in self.result_hist.items():
                    g = f.create_group(f"entry_1/instrument/module_{modno}")
                    g.create_dataset('counts', data=data)
                    g.create_dataset('bins', data=bin_centers)


if __name__ == "__main__":
    path = "/gpfs/exfel/exp/MID/201901/p002542/raw/r0349"
    e = MPIEvalHistogram(path, pixel_hist=False)
    bin_edges = np.linspace(0, 10000, 601, dtype=np.float32)
    e.process(bin_edges, pulse_ids='0:20:1')
    e.hist_to_file(
        '/gpfs/exfel/data/scratch/kamile/calibration_analysis/all_counts_mpi_class.h5')
