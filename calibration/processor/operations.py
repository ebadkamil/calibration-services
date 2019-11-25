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

from karabo_data import DataCollection, by_index, H5File
from ..helpers import pulse_filter, parse_ids


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

        def multihist(chunk, data, bin_edges, ret):
            """Bin counts for many separate 1D arrays along last dimension

            Implementation using numpy.searchsorted() and numpy.add.at()
            to avoid iterating in Python.
            """
            # searchsorted gives the index so x <= bin[i],
            # so we want the upper edge of each bin
            start, end = chunk
            temp = data[:, start:end, :]
            bin_ix = np.searchsorted(bin_edges[1:], temp)

            X, Y, Z = temp.shape
            xgrid, ygrid, zgrid = np.meshgrid(
                np.arange(X), np.arange(Y), np.arange(Z), indexing='ij')

            counts = np.zeros((X, Y, Z, len(bin_edges)), dtype=np.uint32)

            np.add.at(counts, (xgrid, ygrid, zgrid, bin_ix), 1)
            ret[:, start:end, :, :] = counts[..., :-1]

        start = 0
        chunk_size = 32
        chunks = []
        while start < counts.shape[1]:
            chunks.append((start, min(start + chunk_size, counts.shape[1])))
            start += chunk_size

        with ThreadPoolExecutor(max_workers=16) as executor:
            for chunk in chunks:
                executor.submit(
                    multihist, chunk, image, bin_edges, counts)

        print(f"Time taken {time.perf_counter()-t0}")
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


class EvalHistogram:
    """Class to evaluate histogram

    Attributes:
    -----------
    modno: (int) Channel number or module number
    path: (str) Path to Run folder
    dettype: (str) AGIPD, LPD
    pixel_hist: (bool) optional
        Default: False. For pixel wise histogram set it to True"""
    def __init__(self, modno, path, dettype, pixel_hist=False):
        """Initialization"""
        self.histograms = None
        self.mean_image = None

        self.modno = modno
        self.path = path
        self.pixel_hist = pixel_hist
        self.dettype = dettype
        assert self.dettype in ["AGIPD", "LPD"]

    def process(self, bin_edges, pulse_ids=None, workers=None, dark_run=None):
        """Evaluate Histogram and mean image
        Parameters:
        -----------
            bin_edges: (np.ndarray) required
            pulse_ids: str, optional
                Default: all pulses ":"
                For eg. ":" to select all pulses in a train
                "start:stop:step" to select indices with certain step size
                "1,2,3" comma separated pulse index to select specific pulses
                "1,2,3, 5:10" mix of above two
            workers: (int), optional.
                Default: half of total cpus available
                Distribute sequence files over multiple processors
            dark_run: (numpy.ndarray) optional
                dark_dta shape (n_pulses, slow_scan, fast_scan)
                Default: None,
                If provided dark data will be subtracted from images
        """
        self.bin_edges = bin_edges
        self.dark_run = dark_run
        pulse_ids = ":" if pulse_ids is None else pulse_ids
        self.pulses = parse_ids(pulse_ids)

        if not self.path or self.modno not in range(16):
            return

        if workers is None:
            workers = mp.cpu_count() // 2
            print(f"Number of workers: {workers}")
            workers = workers if workers > 0 else 1

        pattern = f"(.+){self.dettype}{self.modno:02d}-S(.+).h5"

        sequences = [osp.join(path, f) for f in os.listdir(self.path)
                     if f.endswith('.h5') and re.match(pattern, f)]

        histograms = []
        images = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for image, hist in executor.map(self._eval, sequences):
                if image is not None and hist is not None:
                    histograms.append(hist)
                    images.append(image)

        if images and histograms:
            self.mean_image = np.mean(np.stack(images), axis=0)
            self.histograms = sum(histograms)

    def _eval(self, seq_file):
        """Histogram over all or individual pixels"""
        if not seq_file:
            return
        run = H5File(seq_file)

        module = [key for key in run.instrument_sources
                  if re.match(r"(.+)/DET/(.+):(.+)", key)]

        if len(module) != 1:
            return

        histogram = 0
        mean_image = 0
        train_counts = 0

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
            train_counts += 1
            counts_pr = []

            if not self.pixel_hist:
                """Evaluate histogram over entire module"""
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

        print("Total ", total.shape)
        if train_counts != 0:
            return mean_image / train_counts, histogram

    def hist_to_file(self, path):
        """Write histograms to H5 File"""
        if all([
            self.histograms is not None,
            self.mean_image is not None,
            path]):

            import h5py

            bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2.0
            with h5py.File(path, "w") as f:
                g = f.create_group(f"entry_1/instrument/module_{self.modno}")
                g.create_dataset('counts', data=self.histograms)
                g.create_dataset('bins', data=bin_centers)
                g.create_dataset('image', data=self.mean_image)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import h5py

    with h5py.File("/home/kamile/calibration_services/dark_run.h5", "r") as f:
        dark_data = f["entry_1/instrument/module_15/data"][:]

    print(f"Shape of dark data: {dark_data.shape}")

    path = "/gpfs/exfel/exp/MID/201931/p900091/raw/r0491"
    counts_file = "/gpfs/exfel/data/scratch/kamile/calibration_analysis/test_pixel.h5"
    modno = 15
    bin_edges = np.linspace(-200, 400, 601)
    pulse_ids = "1:20:2"

    t0 = time.perf_counter()
    e = EvalHistogram(
        modno, path, 'AGIPD', pixel_hist=True)

    e.process(bin_edges, workers=4, pulse_ids=pulse_ids, dark_run=dark_data[0:10, ...])

    print(f"Time taken for histogram Eval.: {time.perf_counter()-t0}")

    e.hist_to_file(counts_file)

    # with h5py.File("/home/kamile/calibration_services/dark_run.h5", "r") as f:
    #     dark_data = f["entry_1/instrument/module_15/data"][:]

    # print(f"Shape of dark data: {dark_data.shape}")

    # path = "/gpfs/exfel/exp/MID/201931/p900091/raw/r0491"
    # module = 15
    # pulse_ids = "1:20:2"
    # bin_edges = np.linspace(-200, 400, 601)
    # bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    # t0 = time.perf_counter()
    # counts = eval_histogram(path, module, bin_edges,
    #                         pulse_ids=pulse_ids,
    #                         dark_run=dark_data[0:10, ...])

    # print(f"Counts shape {counts.shape}")
    # print(f"Time taken for histogram Eval.: {time.perf_counter()-t0}")

    # counts_file = "/gpfs/exfel/data/scratch/kamile/calibration_analysis/pixel_counts.h5"

    # with h5py.File(counts_file, "w") as f:
    #     g = f.create_group(f"entry_1/instrument/module_{module}")
    #     g.create_dataset('counts', data=counts)
    #     g.create_dataset('bins', data=bin_centers)

    # plt.plot(bin_centers, counts[0][64][64])
    # plt.show()