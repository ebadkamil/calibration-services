"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

from enum import IntEnum
from collections import OrderedDict
import os.path as osp
import os
import re
from scipy import constants

import numpy as np
import xarray as xr

from extra_data import DataCollection

from .assembler import ImageAssembler
from .descriptors import MovingAverage, PyFaiAzimuthalIntegrator
from ..helpers import find_proposal, timeit


class PumpProbeMode(IntEnum):
    EVEN_ODD = 1
    ODD_EVEN = 2
    SAME_TRAIN = 3


def detector_data_collection(proposal, run, dettype):
    dettype = dettype.upper()
    assert dettype in ["AGIPD", "LPD", "JUNGFRAU"]
    run_path = find_proposal(proposal, run)
    pattern = f"(.+){dettype}(.+)"

    if dettype == 'JUNGFRAU':
        pattern = f"(.+)JNGFR(.+)"

    files = [os.path.join(run_path, f)
             for f in os.listdir(run_path)
             if f.endswith('.h5') and re.match(pattern, f)]

    if not files:
        return

    data_path = "data.adc" if dettype == "JUNGFRAU" else "image.data"

    run = DataCollection.from_paths(files).select(
        [("*/DET/*", data_path)])

    return run


class PumpProbeAnalysis:

    _pp_mode = OrderedDict({
        "even_odd": PumpProbeMode.EVEN_ODD,
        "odd_even": PumpProbeMode.ODD_EVEN,
        "same_train": PumpProbeMode.SAME_TRAIN,
        })

    def __init__(self, proposal, run, dettype, pp_mode):
        assert pp_mode in PumpProbeAnalysis._pp_mode.keys()
        self.run = detector_data_collection(proposal, run, dettype)
        self.pp_mode = PumpProbeAnalysis._pp_mode[pp_mode]
