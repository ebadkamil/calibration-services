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

from ..helpers import pulse_filter, parse_ids, find_proposal, timeit


@timeit
def dark_offset(proposal, run, module_number, *,
                pulse_ids=None, dettype='AGIPD', eval_std=False):
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
    eval_std: bool
        True: Evaluate standard deviation
        Default: False

    Return
    ------
    out: tuple or ndarray
        if eval_std is set to False
            ndarray: Shape: (n_pulses, ..., slow_scan, fast_scan)
        if eval_std is set to True
            tuple: (mean_image, std)
    """
    path = find_proposal(proposal, run)
    if module_number not in range(16):
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
    std = 0

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

        train_counts += 1

        if eval_std:
            mean_temp = mean_image

        mean_image  = mean_image + (image - mean_image) / train_counts

        if eval_std:
            std = std + (image - mean_temp) * (image - mean_image)

    if train_counts != 0:
        if eval_std:
            return mean_image, np.sqrt(std / train_counts)
        return mean_image

@timeit
def module_roi_intensity(module_number, proposal, run, *,
                         pulse_ids=None, rois=None,
                         dettype='AGIPD', dark_run=None,
                         use_xgm=None):
    """ Evaluate module roi instensity for a given roi

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
        In case of one roi: [x0, x1, y0, y1]
        For multiple rois: [[x0, x1, y0, y1], [x0, x1, y0, y1], ...]
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
    if module_number not in range(16):
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

    run = run.select([(module[0], "image.data")])# for debug .select_trains(by_index[100:200])

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

        roi_images = [image]
        if rois is not None:
            if not isinstance(rois[0], list):
                rois = [rois]
            roi_images = [image[..., x0:x1, y0:y1] for x0, x1, y0, y1 in rois]

        if pulses != [-1]:
            roi_images = [
                img[pulses, ...].astype(np.float32) for img in roi_images] 
        else:
            roi_images = [ 
                img.astype(np.float32) for img in roi_images] 

        if dark_run is not None:
            if not isinstance(dark_run, np.ndarray): # passed as a dict
                try:
                    dark_data = dark_run[str(module_number)]
                except KeyError:
                    dark_data = dark_run[module_number]
            else:
                dark_data = dark_run

            dark_roi_images = [dark_data]

            if rois is not None:
                if not isinstance(rois[0], list):
                    rois = [rois]
                dark_roi_images = [
                    dark_data[..., x0:x1, y0:y1] for x0, x1, y0, y1 in rois]

            if pulses != [-1]:
                dark_roi_images = [img[pulses, ...] for img in dark_roi_images]

            if not all(map(
                lambda x, y: x.shape == y.shape, roi_images, dark_roi_images)):
                raise ValueError("Shapes of image and dark data don't match")

            roi_images = [ 
                roi_images[i] - dark_roi_images[i]
                for i in range(len(roi_images))]

        intensities.append(
            np.stack([np.mean(img, axis=(-1, -2)) for img in roi_images]))
        train_ids.append(tid)

    if not intensities or not train_ids:
        return

    coords = {'trainId': np.array(train_ids)}
    dims = ['trainId', 'rois', 'mem_cells']
    data = xr.DataArray(data=np.stack(intensities), dims=dims, coords=coords)

    if use_xgm is not None:
        files = [f for f in os.listdir(path) if f.endswith('.h5')]
        files = [os.path.join(path, f) for f in fnmatch.filter(files, '*DA*')]

        xgm_data = DataCollection.from_paths(files).get_array(
            use_xgm, "data.intensityTD", extra_dims=['mem_cells'])

        if pulses != [-1]:
            xgm_data = xgm_data[:, pulses]
        else:
            xgm_data = xgm_data[:, 0:data.shape[-1]]

        data, xgm_data = xr.align(data, xgm_data)
        if data.shape[1] == 1:
        # to keep old notebooks happy with no rois dim
            return (data / xgm_data).squeeze(axis=1)

    if data.shape[1] == 1:
        # to keep old notebooks happy with no rois dim
        return data.squeeze(axis=1)
    return data


def split_tiles(module_data):
    half1, half2 = np.split(module_data, 2, axis=-1)
    # Tiles 1-8 (half1) are numbered top to bottom, whereas the array
    # starts at the bottom. So we reverse their order after splitting.
    return np.stack(
        np.split(half1, 8, axis=-2) + np.split(half2, 8, axis=-2))
