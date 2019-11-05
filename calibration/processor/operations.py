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

from karabo_data import DataCollection, by_index
from ..helpers import pulse_filter, parse_ids


def _eval_all(i, bin_edges, path, module_number, pulses, *, dark_run=None):
    """Histogram over all pixel"""

    pattern_sq = f"(.+)AGIPD{module_number:02d}-S{i}(.+)"

    files = [osp.join(path, f) for f in os.listdir(path)
             if f.endswith('.h5') and re.match(pattern_sq, f)]

    if not files:
        return
    # Use H5File instead. Will be just one file
    run = DataCollection.from_paths(files)

    module = [key for key in run.instrument_sources
              if re.match(r"(.+)/DET/(.+):(.+)", key)]

    if len(module) != 1:
        return

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
    return total


def vectorize_para_imp(data, bin_edges, chunk):
    start, end = chunk

    def pixel_histogram(x):
        counts, _ = np.histogram(x, bins=bin_edges)
        return counts
    hist_vectorize = np.vectorize(pixel_histogram, otypes=[np.ndarray])
    hist = hist_vectorize(data[:, start:end, :]).tolist()
    return np.array(hist)


def _eval_pixel(i, bin_edges, path, module_number, pulses, *, dark_run=None):
    """Pixel wise histogram"""

    pattern_sq = f"(.+)AGIPD{module_number:02d}-S{i}(.+)"

    files = [osp.join(path, f) for f in os.listdir(path)
             if f.endswith('.h5') and re.match(pattern_sq, f)]

    if not files:
        return
    # Use H5File instead. Will be just one file
    run = DataCollection.from_paths(files)

    module = [key for key in run.instrument_sources
              if re.match(r"(.+)/DET/(.+):(.+)", key)]

    if len(module) != 1:
        return

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

        counts = np.zeros(
            (len(pulses), 512, 128, len(bin_edges)-1), dtype=np.uint32)

        t0 = time.perf_counter()
        start = 0
        chunk_size = 32
        chunks = []
        while start < counts.shape[1]:
            chunks.append((start, min(start + chunk_size, counts.shape[1])))
            start += chunk_size

        _vectorize_para_imp = partial(vectorize_para_imp, image, bin_edges)

        with ProcessPoolExecutor(max_workers=16) as executor:

            for chunk, ret in zip(chunks, executor.map(
                _vectorize_para_imp, chunks)):
                start, end = chunk
                counts[:, start:end, :, :] = ret

        # print(f"Time taken {time.perf_counter()-t0}")
        total += counts

    print("Total ", total.shape)
    return total


def eval_histogram(path, module_number, bin_edges, *,
                   pulse_ids=None,
                   dark_run=None):

    if not path or module_number not in range(16):
        return

    pattern = f"(.+)AGIPD{module_number:02d}-S(.+).h5"

    sequences = [re.match(pattern, f).group(2) for f in os.listdir(path)
                 if f.endswith('.h5') and re.match(pattern, f)]

    pulse_ids = ":" if pulse_ids is None else pulse_ids
    pulses = parse_ids(pulse_ids)

    _eval_i = partial(
        _eval_pixel,
        bin_edges=bin_edges,
        path=path,
        module_number=module_number,
        pulses=pulses,
        dark_run=dark_run)

    histograms = []
    workers = mp.cpu_count() // 2
    print(f"Number of workers: {workers}")
    workers = workers if workers > 0 else 1

    with ProcessPoolExecutor(max_workers=2) as executor:
        for ret in executor.map(_eval_i, sequences[:2]):
            histograms.append(ret)

    return sum(histograms)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import h5py

    with h5py.File("/home/kamile/calibration_services/dark_run.h5", "r") as f:
        dark_data = f["entry_1/instrument/module_15/data"][:]

    print(f"Shape of dark data: {dark_data.shape}")

    path = "/gpfs/exfel/exp/MID/201931/p900091/raw/r0491"
    module = 15
    pulse_ids = "1:20:2"
    bin_edges = np.linspace(-200, 400, 601)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    t0 = time.perf_counter()
    counts = eval_histogram(path, module, bin_edges,
                            pulse_ids=pulse_ids,
                            dark_run=dark_data[0:10, ...])

    print(f"Counts shape {counts.shape}")
    print(f"Time taken for histogram Eval.: {time.perf_counter()-t0}")

    counts_file = "/gpfs/exfel/data/scratch/kamile/calibration_analysis/pixel_counts.h5"

    with h5py.File(counts_file, "w") as f:
        g = f.create_group(f"entry_1/instrument/module_{module}")
        g.create_dataset('counts', data=counts)
        g.create_dataset('bins', data=bin_centers)

    plt.plot(bin_centers, counts[0][64][64])
    plt.show()