"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import fnmatch
import multiprocessing as mp
import os.path as osp
import os
import re
import time

import h5py
import numpy as np
import xarray as xr

from karabo_data import DataCollection, by_index, H5File

from ..helpers import pulse_filter, parse_ids, find_proposal


def dark_offset(proposal, run, module_number, *,
                pulse_ids=None, dettype='AGIPD'):
    """ Process Dark data

    Parameters
    ----------
    module_number: int
        Channel number between 0, 15
    proposal: str, int
        A proposal number, such as 2012, '2012', 'p002012', or a path such as
        '/gpfs/exfel/exp/SPB/201701/p002012'.
    run: str, int
        A run number such as 243, '243' or 'r0243'.
    pulse_ids: str
        For eg. ":" to select all pulses in a train
                "start:stop:step" to select indices with certain step size
                "1,2,3" comma separated pulse index to select specific pulses
                "1,2,3, 5:10" mix of above two
        Default: all pulses ":"
    dettype: str
        "AGIPD", "LPD"

    Return
    ------
    out: ndarray
        Shape: (n_pulses, ..., slow_scan, fast_scan)
    """
    path = find_proposal(proposal, run)
    if module_number not in range(17):
        print(f"Module number should be in range 0-15, got {module_number}")
        return

    pattern = f"(.+){dettype}{module_number:02d}(.+)"

    files = [os.path.join(path, f) for f in os.listdir(path)
             if f.endswith('.h5') and re.match(pattern, f)]

    if not files:
        return

    run = DataCollection.from_paths(files)

    module = [key for key in run.instrument_sources
              if re.match(r"(.+)/DET/(.+):(.+)", key)]

    if len(module) != 1:
        return

    run = run.select([(module[0], "image.data")])

    pulse_ids = ":" if pulse_ids is None else pulse_ids
    pulses = parse_ids(pulse_ids)

    mean_image = 0
    train_counts = 0
    for tid, data in run.trains():
        if dettype == 'LPD':
            image = np.squeeze(
                data[module[0]]["image.data"], axis=1) # (pulses, 1, ss, fs)
        else:
            image = data[module[0]]["image.data"][:, 0, ...]

        if image.shape[0] == 0:
            continue

        if pulses != [-1]:
            image = image[pulses, ...].astype(np.float32)
        else:
            image = image.astype(np.float32)

        mean_image += image
        train_counts += 1

    if train_counts != 0:
        return mean_image / train_counts


def _eval(seq, pulses, dettype, rois=None, dark_run=None):
    run = H5File(seq)
    module = [key for key in run.instrument_sources
              if re.match(r"(.+)/DET/(.+):(.+)", key)]

    if len(module) != 1:
        return

    run = run.select([(module[0], "image.data")])
    partial_intensities = []
    partial_train_ids = []
    for tid, data in run.trains():
        if dettype == 'LPD':
            image = np.squeeze(
                data[module[0]]["image.data"], axis=1) # (pulses, 1, ss, fs)
        else:
            image = data[module[0]]["image.data"][:, 0, ...]
        if image.shape[0] == 0:
            continue

        if rois is not None:
            x0, x1, y0, y1 = rois
            image = image[..., x0:x1, y0:y1]

        if pulses != [-1]:
            image = image[pulses, ...].astype(np.float32)
        else:
            image = image.astype(np.float32)
        if dark_run is not None:
            if pulses != [-1]:
                dark_data = dark_run[pulses, ...]
            else:
                dark_data = dark_run

            if rois is not None:
                x0, x1, y0, y1 = rois
                dark_data = dark_data[..., x0:x1, y0:y1]

            if image.shape == dark_data.shape:
                image -= dark_data
            else:
                raise ValueError(''
                    f"Different data shapes, dark_data: {dark_data.shape}"
                    f" Run data: {image.shape}")

        partial_intensities.append(np.mean(image, axis=(-1, -2)))
        partial_train_ids.append(tid)

    return partial_intensities, partial_train_ids

def module_roi_intensity(module_number, proposal, run, *,
                         pulse_ids=None, rois=None,
                         dettype='AGIPD', dark_run=None,
                         use_xgm=None):
    """ Process Dark data

    Parameters
    ----------
    module_number: str, int
        Channel number between 0, 15
    proposal: str, int
        A proposal number, such as 2012, '2012', 'p002012', or a path such as
        '/gpfs/exfel/exp/SPB/201701/p002012'.
    run: str, int
        A run number such as 243, '243' or 'r0243'.
    pulse_ids: str
        For eg. ":" to select all pulses in a train
                "start:stop:step" to select indices with certain step size
                "1,2,3" comma separated pulse index to select specific pulses
                "1,2,3, 5:10" mix of above two
        Default: all pulses ":"
    rois: list
        [x0, x1, y0, y1]
    dettype: str
        "AGIPD", "LPD"
    dark_run: (numpy.ndarray) or dict optional
        dark_data shape (n_pulses, slow_scan, fast_scan)
        dark_run[module_number] of shape (n_pulses, slow_scan, fast_scan)
        Default: None,
        If provided dark data will be subtracted from images
    use_xgm: str
        XGM source name to be used to normalize data.
        For eg: "SPB_XTD9_XGM/DOOCS/MAIN:output"

    Return
    ------
    out: xarray
        The first axis of the returned data will be labelled with the trainId
        Shape of numpy array: (n_trains, n_pulses)
    """
    if not isinstance(module_number, int):
        module_number = int(module_number)

    path = find_proposal(proposal, run)
    if module_number not in range(17):
        print(f"Module number should be in range 0-15, got {module_number}")
        return

    # pattern = f"(.+){dettype}{module_number:02d}-S(.+).h5"

    # sequences = [osp.join(path, f) for f in os.listdir(path)
    #              if f.endswith('.h5') and re.match(pattern, f)]

    # pulse_ids = ":" if pulse_ids is None else pulse_ids
    # pulses = parse_ids(pulse_ids)

    # evaluate = partial(_eval,
    #     pulses=pulses, dettype=dettype, dark_run=dark_run, rois=rois)

    # intensities = []
    # train_ids = []

    # with ProcessPoolExecutor(max_workers=2) as executor:
    #     for partial_intensities, partial_train_ids in executor.map(
    #         evaluate, sorted(sequences)):
    #         if partial_intensities and partial_train_ids:
    #             intensities.extend(partial_intensities)
    #             train_ids.extend(partial_train_ids)

    # if not intensities or not train_ids:
    #     return

    # coords = {'trainId': np.array(train_ids)}
    # dims = ['trainId', 'dim_0']
    # data = xr.DataArray(np.stack(intensities), dims=dims, coords=coords)

    # return data
    pattern = f"(.+){dettype}{module_number:02d}(.+)"

    files = [os.path.join(path, f) for f in os.listdir(path)
             if f.endswith('.h5') and re.match(pattern, f)]

    if not files:
        return

    run = DataCollection.from_paths(files)

    module = [key for key in run.instrument_sources
              if re.match(r"(.+)/DET/(.+):(.+)", key)]

    if len(module) != 1:
        return

    run = run.select([(module[0], "image.data")])

    pulse_ids = ":" if pulse_ids is None else pulse_ids
    pulses = parse_ids(pulse_ids)

    intensities = []
    train_ids = []

    for tid, data in run.trains():
        if dettype == 'LPD':
            image = np.squeeze(
                data[module[0]]["image.data"], axis=1) # (pulses, 1, ss, fs)
        else:
            image = data[module[0]]["image.data"][:, 0, ...]

        if image.shape[0] == 0:
            continue

        if rois is not None:
            x0, x1, y0, y1 = rois
            image = image[..., x0:x1, y0:y1]

        if pulses != [-1]:
            image = image[pulses, ...].astype(np.float32)
        else:
            image = image.astype(np.float32)

        if dark_run is not None:
            if not isinstance(dark_run, np.ndarray): # passed as a dict
                dark_data = dark_run[str(module_number)]
            else:
                dark_data = dark_run

            if pulses != [-1]:
                dark_data = dark_data[pulses, ...]

            if rois is not None:
                x0, x1, y0, y1 = rois
                dark_data = dark_data[..., x0:x1, y0:y1]

            if image.shape == dark_data.shape:
                image -= dark_data
            else:
                raise ValueError(
                    f"Different data shapes, dark_data: {dark_data.shape}"
                    f" Run data: {image.shape}")

        intensities.append(np.mean(image, axis=(-1, -2)))
        train_ids.append(tid)

    if not intensities or not train_ids:
        return

    coords = {'trainId': np.array(train_ids)}
    dims = ['trainId', 'dim_0']
    data = xr.DataArray(np.stack(intensities), dims=dims, coords=coords)

    if use_xgm is not None:
        files = [f for f in os.listdir(path) if f.endswith('.h5')]
        files = [os.path.join(path, f) for f in fnmatch.filter(files, '*DA*')]

        xgm_data = DataCollection.from_paths(files).get_array(
            use_xgm, "data.intensityTD")

        if pulses != [-1]:
            xgm_data = xgm_data[:, pulses]
        else:
            xgm_data = xgm_data[:, 0:data.shape[0]]

        data, xgm_data = xr.align(data, xgm_data)
        return data / xgm_data

    return data
