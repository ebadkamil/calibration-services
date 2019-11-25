"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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

from .MPI import distribute
from ..helpers import pulse_filter, parse_ids

comm = MPI.COMM_WORLD


def eval_histogram(path, bin_edges, pulses, sequences, *, dark_run=None):

    histogram = np.zeros(
        (len(pulses),  bin_edges.shape[0] - 1), dtype=np.int64)

    if not sequences:
        return None, histogram

    modno, files = zip(*sequences)

    run = DataCollection.from_paths(list(files))

    module = [key for key in run.instrument_sources
              if re.match(r"(.+)/DET/(.+):(.+)", key)]

    if len(module) != 1:
        raise ValueError("More than one module found")

    num_trains = 0
    for tid, data in run.trains(devices=[(module[0], "image.data")],
                                require_all=True):

        image = data[module[0]]["image.data"][:, 0, ...]
        if image.shape[0] == 0:
            continue

        if pulses != [-1]:
            image = image[pulses, ...].astype(np.float32)
        else:
            image = image.astype(np.float32)

        if dark_run is not None:
            dark_data = dark_run[:, 0, ...]
            if image.shape == dark_data.shape:
                image -= dark_data
            else:
                raise ValueError(
                    f"Different data shapes, dark_data: {dark_data.shape}"
                    f" Run data: {image.shape}")

        counts_pr = []

        def _eval_stat(pulse):
            counts, _ = np.histogram(
                image[pulse, ...].ravel(), bins=bin_edges)
            return counts

        with ThreadPoolExecutor(max_workers=5) as executor:
            for ret in executor.map(_eval_stat, range(image.shape[0])):
                counts_pr.append(ret)

        histogram += np.stack(counts_pr)

    return modno[0], histogram


if __name__ == "__main__":

    # path = "/gpfs/exfel/exp/MID/201931/p900091/raw/r0491"
    path = "/gpfs/exfel/exp/MID/201901/p002542/raw/r0349"
    dark_data_file = "/home/kamile/calibration_services/dark_run.h5"
    pulse_ids = "1:20:2"
    pulses = parse_ids(pulse_ids)

    bin_edges = np.linspace(-200, 400, 601, dtype=np.float32)

    channels, local_sequences = distribute(path)
    print(local_sequences)

    dark_data_shape = None

    if comm.rank == 0:
        with h5py.File(dark_data_file, "r") as f:
            dark_data = f["entry_1/instrument/module_15/data"][:].astype(
                np.float32)
        dark_data_shape = dark_data.shape
        print(f"Shape of dark_data {dark_data_shape}")

    dark_data_shape = comm.bcast(dark_data_shape, root=0)

    if comm.rank != 0:
        dark_data = np.empty(dark_data_shape, dtype=np.float32)

    comm.Bcast(dark_data, root=0)

    modno, hist = eval_histogram(
        path, bin_edges, pulses, local_sequences,
        dark_run=dark_data[0:10, ...])

    if comm.rank == 0:
        result = {mod: 0 for mod in channels}
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

        if modno is not None:
            result[modno] += hist

        for source in range(1, comm.size):
            mod = comm.recv(source=source)
            histogram = np.empty(
                (len(pulses),) + bin_centers.shape, dtype=np.int64)
            comm.Recv(histogram, source=source)
            if mod is not None:
                result[mod] += histogram

        file = "/gpfs/exfel/data/scratch/kamile/calibration_analysis/all_counts.h5"
        with h5py.File(file, "w") as f:
            for modno, data in result.items():
                g = f.create_group(f"entry_1/instrument/module_{modno}")
                g.create_dataset('counts', data=result[modno])
                g.create_dataset('bins', data=bin_centers)

    else:
        comm.send(modno, dest=0)
        comm.Send(hist, dest=0)
