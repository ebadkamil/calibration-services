"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
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


def eval_histogram(path, bin_edges, pulses, sequences, *, dark_run=None):

    modno, files = zip(*sequences)

    if not files:
        return modno[0], None

    files = [osp.join(path, f) for f in files]

    run = DataCollection.from_paths(list(files))

    module = [key for key in run.instrument_sources
              if re.match(r"(.+)/DET/(.+):(.+)", key)]

    if len(module) != 1:
        return modno[0], None

    total = 0

    for tid, data in run.trains(devices=[(module[0], "image.data")],
                                require_all=True):

        image = data[module[0]]["image.data"][:, 0, ...]
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
        total += np.stack(counts_pr)

    print("Total ", total.shape)
    return modno[0], total


if __name__ == "__main__":
    import h5py

    path = "/gpfs/exfel/exp/MID/201931/p900091/raw/r0491"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    index = None
    modules = None
    module_sequences = None
    pulses = None
    bin_edges_shape = None
    dark_data_shape = None

    if rank == 0:
        pulse_ids = "1:20:2"
        bin_edges = np.linspace(-200, 400, 601, dtype=np.float32)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        bin_edges_shape = bin_edges.shape

        pulses = parse_ids(pulse_ids)

        with h5py.File("/home/kamile/calibration_services/dark_run.h5", "r") as f:
            dark_data = f["entry_1/instrument/module_15/data"][:].astype(
                np.float32)

        dark_data_shape = dark_data.shape

        pattern = f"(.+)AGIPD(.+)-S(.+).h5"

        sequences = [(
            re.match(pattern, f).group(2),
            f) for f in os.listdir(path)
            if f.endswith('.h5') and re.match(pattern, f)]

        module_sequences = []
        module_channels = []
        for key, group in groupby(sorted(sequences), lambda x: x[0]):
            module_sequences.append(list(group))
            module_channels.append(key)

        if size % len(module_sequences) != 0:
            print(f"Use multiple of {len(module_numbers)} processes")
            sys.exit()

        index = size // len(module_sequences)

    bin_edges_shape = comm.bcast(bin_edges_shape, root=0)
    dark_data_shape = comm.bcast(dark_data_shape, root=0)

    index = comm.bcast(index, root=0)
    modules = comm.bcast(module_sequences, root=0)
    pulses = comm.bcast(pulses, root=0)

    if rank != 0:
        dark_data = np.empty(dark_data_shape, dtype=np.float32)
        bin_edges = np.empty(bin_edges_shape, dtype=np.float32)

    comm.Bcast(dark_data, root=0)
    comm.Bcast(bin_edges, root=0)

    local_modno = rank // index
    temp = rank % index
    sequence_cuts = [int(
        len(modules[local_modno]) * i / index) for i in range(index + 1)]

    chunks = list(zip(sequence_cuts[:-1], sequence_cuts[1:]))

    local_sequences = modules[local_modno][chunks[temp][0]:chunks[temp][1]]

    modno, hist = eval_histogram(
        path, bin_edges, pulses, local_sequences,
        dark_run=dark_data[0:10, ...])

    if rank == 0:
        result = {}
        for mod in module_channels:
            result[mod] = 0

        result[modno] += hist
        for source in range(1, size):
            mod = comm.recv(source=source)
            histogram = np.empty(
                (len(pulses),) + bin_edges_shape, dtype=np.int64)
            comm.Recv(histogram, source=source)
            result[mod] += histogram

        file = "/gpfs/exfel/data/scratch/kamile/calibration_analysis/all_counts.h5"
        with h5py.File(file, "w") as f:
            for modno, data in result.items():
                g = f.create_group(f"entry_1/instrument/module_{modno}")
                g.create_dataset('data', data=result[modno])

    else:
        comm.send(modno, dest=0)
        comm.Send(hist, dest=0)
