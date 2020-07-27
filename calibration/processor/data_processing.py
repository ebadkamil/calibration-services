"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

import os.path as osp
import os
import re
import numpy as np
from scipy.optimize import curve_fit

from extra_data import DataCollection, by_index

from ..helpers import pulse_filter, parse_ids

def DataProcessing(module_number, path, *,
                   train_index=None, pulse_ids=None,
                   rois=None, operation=None,
                   dark_run=None):
    """ Process Data

    Parameters
    ----------
    module_number: int
        Channel number between 0, 15
    path: str
        Path to Run folder
    train_index: extra_data (by_index)
        Default (all trains by_index[:])
    pulse_ids: str
        For eg. ":" to select all pulses in a train
                "start:stop:step" to select indices with certain step size
                "1,2,3" comma separated pulse index to select specific pulses
                "1,2,3, 5:10" mix of above two
        Default: all pulses ":"
    rois: extra_data slice constructor by_index
        Select ROI of image data. For eg. by_index[..., 0:128, 0:64]
        See extra_data method: `get_array`

    operation: function
        For eg. functools.partial(np.mean, axis=0) to take mean over trains
    dark_run: ndarray
        dark_data to subtract

    Return
    ------
    out: ndarray
        Shape:  operation -> (n_trains, n_pulses, ..., slow_scan, fast_scan)
    """

    if not path or module_number not in range(16):
        return

    pattern = f"(.+)AGIPD{module_number:02d}(.+)"

    files = [osp.join(path, f) for f in os.listdir(path)
             if f.endswith('.h5') and re.match(pattern, f)]

    if not files:
        return

    run = DataCollection.from_paths(files)

    module = [key for key in run.instrument_sources
              if re.match(r"(.+)/DET/(.+):(.+)", key)]

    if len(module) != 1:
        return

    pulse_ids = ":" if pulse_ids is None else pulse_ids
    rois = by_index[..., :, :] if rois is None else rois
    train_index = by_index[:] if train_index is None else train_index

    run = run.select([(module[0], "image.data")]).select_trains(train_index)

    counts = run.get_data_counts(module[0], "image.data")
    pulses = pulse_filter(pulse_ids, counts[counts != 0])
    # TODO: Not a good way to load all data. Will replace with looping over
    # trains while performing dark average. For getting dark subtracted
    # train (one train or a range of trains) data, have to think about
    # better way. (`next(run.trains())`)
    # Looping over trains for average does not improve speed.

    data = run.get_array(module[0], "image.data",
                         roi=rois).values[pulses, ...].astype(np.float32)

    if data.size == 0:
        raise ValueError(f"Empty data array for train_index {train_index}")

    if dark_run is not None:

        if dark_run.shape == data.shape[1:]:
            data -= dark_run
        else:
            raise ValueError(
                f"Different data shapes, dark_data: {dark_module.shape}"
                f" Run data: {data.shape[1:]}")

    if operation is not None:
        return operation(data)
    else:
        return data


class Statistics:
    def __init__(self):
        self.bin_centers = None
        self.bin_counts = None
        self.filtered = None
        self.peaks = None


class DataModel:

    class DarkData:
        def __init__(self):
            self.image = None
            self.st = Statistics()

    class ProcessedData:
        def __init__(self):
            self.image = None
            self.st = Statistics()

    def __init__(self):
        self.dark_data = self.DarkData()
        self.proc_data = self.ProcessedData()


def eval_statistics(image, bins=None):
    # img = np.copy(image) Unnecessary

    bins = 100 if bins is None else bins
    counts, edges = np.histogram(image.ravel(), bins)
    centers = (edges[1:] + edges[:-1]) / 2.0
    return centers, counts


def gaussian(x, *params):
    num_gaussians = int(len(params) / 3)
    A = params[:num_gaussians]
    w = params[num_gaussians:2*num_gaussians]
    c = params[2*num_gaussians:3*num_gaussians]
    y = sum([A[i]*np.exp(-(x-c[i])**2./(w[i])) for i in range(num_gaussians)])
    return y


def gauss_fit(xdata, ydata, params):
    """
    Parameters:
    ----------
    xdata: 1d array
    ydata: 1d array
    params: list
        [A1, A2, A3, ..., S1, S2, S3, ..., P1, P2, P3...]
        A: Amplitude of Gaussian
        S: Width of Gaussian
        P: Centre of Gaussian

    Return
    ------
    fit_data: 1d array
    popt: fit params
    perr: np.sqrt(np.diag(pcov))
        standard error in fit params
    """

    popt, pcov = curve_fit(gaussian, xdata, ydata, p0=params)
    return (gaussian(xdata, *popt), popt, np.sqrt(np.diag(pcov)))

