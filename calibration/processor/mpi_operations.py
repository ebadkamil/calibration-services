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
from itertools import groupby

from ..helpers import pulse_filter, parse_ids
from .MPI import distribute


class MPIEvalHistogram:
     """MPI Class to evaluate histogram

    Attributes:
    -----------
    path: (str) Path to Run folder
    dettype: (str) AGIPD, LPD
    pixel_hist: (bool) optional
        Default: False. For pixel wise histogram set it to True
    dark_run: (numpy.ndarray) optional
        dark_dta shape (n_pulses, slow_scan, fast_scan)
        Default: None,
        If provided dark data will be subtracted from images"""

    def __init__(self, path,
                 pixel_hist=False, comm=MPI.COMM_WORLD, dark_run=None):
        """Initialization"""
        self.comm = comm
        self.path = path
        self.pixel_hist = pixel_hist
        self.dark_run = dark_run

        self.modno = None
        self.histogram = None

    def process(self, bin_edges, pulse_ids=None, workers=None):
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

        self.eval_histogram()
        self.gather()

    def gather(self):

        if self.comm.rank == 0:
            self.result = {mod: 0 for mod in self.channels}

            if self.modno is not None:
                self.result[self.modno] += self.histogram

            for source in range(1, self.comm.size):
                mod = self.comm.recv(source=source)
                if mod is not None:
                    shape = self.comm.recv(source=source)
                    dtype = self.comm.recv(source=source)
                    print("Recv ", source, shape, dtype)
                    histogram = np.empty(shape, dtype=dtype)
                    self.comm.Recv(histogram, source=source)
                else:
                    histogram = self.comm.recv(source=source)

                if mod is not None and histogram is not None:
                    self.result[mod] += histogram
        else:
            self.comm.send(self.modno, dest=0)

            if self.modno is not None:
                shape = self.histogram.shape
                dtype = self.histogram.dtype
                print("Send ", self.comm.rank, shape, dtype)
                self.comm.send(shape, dest=0)
                self.comm.send(dtype, dest=0)
                self.comm.Send(self.histogram, dest=0)
            else:
                self.comm.send(self.histogram, dest=0)

    def eval_histogram(self):

        if not self.local_sequences:
            return

        modno, files = zip(*self.local_sequences)

        run = DataCollection.from_paths(list(files)).select_trains(by_index[:20])

        module = [key for key in run.instrument_sources
                  if re.match(r"(.+)/DET/(.+):(.+)", key)]

        if len(module) != 1:
            raise ValueError("More than one module found")

        num_trains = 0
        histogram = 0

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

                    counts = np.zeros((X, Y, Z, len(bin_edges)), dtype=np.uint32)

                    np.add.at(counts, (xgrid, ygrid, zgrid, bin_ix), 1)
                    return counts[..., :-1]

                counts = multihist(image, self.bin_edges)
                histogram += counts

            num_trains += 1

        if num_trains != 0:
            self.modno = modno[0]
            self.histogram = histogram

    def hist_to_file(self, path):
        if path and self.comm.rank == 0:
            bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2.0

            with h5py.File(path, "w") as f:
                for modno, data in self.result.items():
                    g = f.create_group(f"entry_1/instrument/module_{modno}")
                    g.create_dataset('counts', data=data)
                    g.create_dataset('bins', data=bin_centers)


if __name__ == "__main__":
    path = "/gpfs/exfel/exp/MID/201901/p002542/raw/r0349"
    e = MPIEvalHistogram(path, pixel_hist=True)
    bin_edges = np.linspace(0, 10000, 601, dtype=np.float32)
    e.process(bin_edges, pulse_ids='0:20:1')
    e.hist_to_file('/gpfs/exfel/data/scratch/kamile/calibration_analysis/all_counts_mpi_class.h5')
