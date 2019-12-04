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
import time

import h5py
import numpy as np

from iminuit import Minuit
from karabo_data import DataCollection, by_index, H5File
from ..helpers import pulse_filter, parse_ids, find_proposal


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
        self.bin_edges = None
        self.mean_image = None
        self.fit_params = None

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

        sequences = [osp.join(self.path, f) for f in os.listdir(self.path)
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
                dark_data = self.dark_run
                if image.shape == dark_data.shape:
                    image -= dark_data
                else:
                    raise ValueError(
                        f"Different data shapes, dark_data: {dark_data.shape}"
                        f" Run data: {image.shape}")

            mean_image += image
            train_counts += 1

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

        print("Total ", histogram.shape)
        if train_counts != 0:
            return mean_image / train_counts, histogram

    def hist_to_file(self, path):
        """Write histograms to H5 File"""
        if all([
            self.histograms is not None,
            self.mean_image is not None,
            path]):

            bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2.0
            with h5py.File(path, "w") as f:
                g = f.create_group(f"entry_1/instrument/module_{self.modno}")
                g.create_dataset('counts', data=self.histograms)
                g.create_dataset('bins', data=bin_centers)
                g.create_dataset('image', data=self.mean_image)

    def fit_histogram(self, init_params, bounds_params,
                      from_file=None, threshold=(-50, 120)):

        histogram = self.histograms
        bin_edges = self.bin_edges

        if bin_edges is not None:
            bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2.0

        self.init_params = init_params
        self.bounds_params = bounds_params

        if from_file is not None:
            with h5py.File(from_file, "r") as f:
                bin_centers = \
                    f[f"entry_1/instrument/module_{self.modno}/bins"][:]
                histogram = \
                    f[f"entry_1/instrument/module_{self.modno}/counts"][:]


        if histogram is None:
            return
        low, high = threshold
        idx = (bin_centers > low) & (bin_centers < high)

        hist_for_each = np.split(histogram.flatten(),
                                 np.product(histogram.shape[:-1]))

        map_fitting = partial(self._fitting, idx, bin_centers)

        with ProcessPoolExecutor(max_workers=20) as executor:
            ret = executor.map(map_fitting, hist_for_each)

        self.fit_params = np.array(
            list(ret)).reshape(
            histogram.shape[:-1]+(2*len(self.init_params)+1,))

    def _fitting(self, idx, bin_centers, histogram):

        def gaussian(x, *params):
            num_gaussians = int(len(params) / 3)
            A = params[:num_gaussians]
            w = params[num_gaussians:2*num_gaussians]
            c = params[2*num_gaussians:3*num_gaussians]
            y = sum(
                [A[i]*np.exp(-(x-c[i])**2./(w[i])) for i in range(num_gaussians)])

            return y

        def least_squares_np(xdata, ydata,  params):
            y = gaussian(xdata, *params)
            return np.sum((ydata - y) ** 2)

        least_sq = partial(
            least_squares_np,
            bin_centers[idx],
            histogram[idx])

        m = Minuit.from_array_func(
            least_sq,
            self.init_params,
            error=0.1,
            errordef=1,
            limit=tuple(self.bounds_params))


        minuit_res = m.migrad()

        return np.concatenate(
            (m.np_values(),
             m.np_errors(),
             np.array([m.get_fmin().is_valid])))

    def fit_params_to_file(self, path):
        """Write fit params to H5 File"""
        if all([self.fit_params is not None, path]):
            with h5py.File(path, "w") as f:
                g = f.create_group(f"entry_1/instrument/module_{self.modno}")
                g.create_dataset('fit_params', data=self.fit_params)


if __name__ == "__main__":

    path = "/gpfs/exfel/exp/MID/201931/p900091/raw/r0491"
    counts_file = "/gpfs/exfel/data/scratch/kamile/calibration_analysis/test_pixel.h5"
    fit_file = "/gpfs/exfel/data/scratch/kamile/calibration_analysis/fit.h5"
    modno = 15
    bin_edges = np.linspace(-200, 400, 601)
    pulse_ids = "1:24:2"

    dark_file = os.path.join(
        "/gpfs/exfel/data/scratch/kamile/batch",
        f"dark_module_{modno}.h5")

    histogram_file = os.path.join(
        "/gpfs/exfel/data/scratch/kamile/batch",
        f"data_module_{modno}.h5")

    with h5py.File(dark_file, "r") as f:
        dark_data = f[f"entry_1/instrument/module_{modno}/image"][:]

    print(f"Shape of dark data: {dark_data.shape}")

    t0 = time.perf_counter()
    e = EvalHistogram(
        modno, path, 'AGIPD', pixel_hist=True)

    # e.process(bin_edges, workers=5, pulse_ids=pulse_ids, dark_run=dark_data)
    # e.hist_to_file(counts_file)

    params = [100, 70, 50, 10, 10, 10, -25, 25, 70]
    bounds_minuit = [(0, None), (0, None), (0, None),
                     (0, None), (0, None), (0, None),
                     (-50, 0), (0, 50), (40, 100)]

    e.fit_histogram(params, bounds_minuit, from_file=histogram_file)
    e.fit_params_to_file(fit_file)
    print(f"Time taken for histogram Eval.: {time.perf_counter()-t0}")