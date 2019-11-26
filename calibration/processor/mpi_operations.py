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

    def __init__(self, path,
                 pixel_hist=False, comm=MPI.COMM_WORLD, dark_run=None):
        self.comm = comm
        self.path = path
        self.pixel_hist = pixel_hist
        self.dark_run = dark_run

    def process(self, bin_edges, pulse_ids=None, workers=None):
        self.bin_edges = bin_edges
        pulse_ids = ":" if pulse_ids is None else pulse_ids
        self.pulses = parse_ids(pulse_ids)

        self.channels, self.local_sequences = distribute(
            self.path, comm=self.comm)

        print(self.local_sequences)

        self.modno, self.histogram = self.eval_histogram()
        self.gather()

    def gather(self):

        if self.comm.rank == 0:
            self.result = {mod: 0 for mod in self.channels}
            self.bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2.0

            if self.modno is not None:
                self.result[self.modno] += self.histogram

            for source in range(1, self.comm.size):
                mod = self.comm.recv(source=source)
                if mod is not None:
                    histogram = np.empty(
                        (len(self.pulses),) + bin_centers.shape, dtype=np.int64)
                    self.comm.Recv(histogram, source=source)
                else:
                    histogram = self.comm.recv(source=source)

                if mod is not None and histogram is not None:
                    self.result[mod] += histogram
        else:
            self.comm.send(self.modno, dest=0)
            if self.modno is not None:
                self.comm.Send(self.histogram, dest=0)
            else:
                self.comm.send(self.histogram, dest=0)

    def eval_histogram(self):

        if not self.local_sequences:
            return None, None

        modno, files = zip(*self.local_sequences)

        run = DataCollection.from_paths(list(files))

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
                    (len(self.pulses),
                    512,
                    128,
                    len(self.bin_edges)-1),
                    dtype=np.uint32)

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
                            multihist, chunk, image, self.bin_edges, counts)

                histogram += counts

            num_trains += 1
        if num_trains != 0:
            return modno[0], histogram
        else:
            return None, None

    def hist_to_file(self, path):
        if path and self.comm.rank == 0:
            with h5py.File(path, "w") as f:
                for modno, data in self.result.items():
                    g = f.create_group(f"entry_1/instrument/module_{modno}")
                    g.create_dataset('counts', data=data)
                    g.create_dataset('bins', data=self.bin_centers)


if __name__ == "__main__":
    path = "/gpfs/exfel/exp/MID/201901/p002542/raw/r0349"
    e = MPIEvalHistogram(path)
    bin_edges = np.linspace(0, 10000, 1001, dtype=np.float32)
    e.process(bin_edges, pulse_ids='0:20:1')
