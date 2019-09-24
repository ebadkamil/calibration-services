import os.path as osp
import os
import re
import numpy as np

from ..helpers import pulse_filter, parse_ids
from karabo_data import DataCollection, by_index


def DataProcessing(module_number, path, *,
                   train_index=None, pulse_ids=None,
                   rois=None, operation=None,
                   dark_run=None):
    """ Process Data

    Parameters
    ----------
    module_number: int
        Channel number between 0, 16
    path: str
        Path to Run folder
    train_index: karabo_data (by_index)
        Default (all trains by_index[:])
    pulse_ids: str
        For eg. ":" to select all pulses in a train
                "start:stop:step" to select indices with certain step size
                "1,2,3" comma separated pulse index to select specific pulses
                "1,2,3, 5:10" mix of above two
        Default: all pulses ":"
    rois: karabo_data slice constructor by_index
        Select ROI of image data. For eg. by_index[..., 0:128, 0:64]
        See karabo_data method: `get_array`

    operation: function
        For eg. functools.partial(np.mean, axis=0) to take mean over trains
    dark_run: nd.array
        dark_data to subtract

    Return
    ------
    out: ndarray
        Shape:  operation -> (n_trains, n_pulses, ..., slow_scan, fast_scan)
    """

    if operation is None or not path or module_number not in range(16):
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

    data = run.get_array(module[0], "image.data",
                         roi=rois).values[pulses, ...].astype(np.float32)

    if dark_run is not None:
        dark_module = dark_run[module_number]

        if dark_module.shape == data.shape[1:]:
            data -= dark_module
        else:
            print(f"Different data shapes, dark_data: {dark_module.shape}"
                  f" Run data: {data.shape[1:]}")

    return operation(data)
