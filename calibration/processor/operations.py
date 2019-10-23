"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import os.path as osp
import os
import re
import numpy as np

from karabo_data import DataCollection, by_index
from ..helpers import pulse_filter, parse_ids

def _eval(i, bin_edges, path, module_number, pulses, *, dark_run=None):
    pattern_sq = f"(.+)AGIPD{module_number:02d}-S{i}(.+)"

    files = [osp.join(path, f) for f in os.listdir(path)
             if f.endswith('.h5') and re.match(pattern_sq, f)]

    if not files:
        return

    run = DataCollection.from_paths(files)

    module = [key for key in run.instrument_sources
              if re.match(r"(.+)/DET/(.+):(.+)", key)]

    if len(module) != 1:
        return

    total = np.zeros((len(pulses), len(bin_edges) - 1))

    out = np.zeros((len(pulses), 2, 512, 128), dtype=np.float32)

    for tid, data in run.trains(devices=[(module[0], "image.data")],
        require_all=True):

        image = data[module[0]]["image.data"]
        if pulses != [-1]:
            image = image[pulses, ...].astype(np.float32)
        else:
            image = image.astype(np.float32)

        if dark_run is not None:
            if image.shape == dark_run.shape:
                image -= dark_run

        counts_pr=[]

        def _eval_stat(pulse):
            counts, _ = np.histogram(
                image[pulse, 0, ...].ravel(), bins=bin_edges)
            return counts

        with ThreadPoolExecutor(max_workers=10) as executor:
            for ret in executor.map(_eval_stat, range(image.shape[0])):
                counts_pr.append(ret)

        total += np.stack(counts_pr)

    print(total.shape)
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
        _eval,
        bin_edges=bin_edges,
        path=path,
        module_number=module_number,
        pulses=pulses,
        dark_run=dark_run)

    histograms = []
    with ProcessPoolExecutor(max_workers=len(sequences[:4])) as executor:
        for ret in executor.map(_eval_i, sequences[4:8]):
            histograms.append(ret)

    return sum(histograms)


if __name__ == "__main__":
    path = "/gpfs/exfel/exp/MID/201931/p900091/raw/r0491"
    module = 15
    pulse_ids = "1:250:2"
    bin_edges = np.linspace(-200, 400, 601)
    counts = eval_histogram(path, module, bin_edges, 
                            dark_run=dark_data, pulse_ids=pulse_ids)
