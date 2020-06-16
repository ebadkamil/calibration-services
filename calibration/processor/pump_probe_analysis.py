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


class AnalysisType(IntEnum):
    ROI = 1
    AZIMUTHAL = 2


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

    _analysis_type = OrderedDict({
        "roi": AnalysisType.ROI,
        "azimuthal": AnalysisType.AZIMUTHAL,
        })

    def __init__(self, proposal, run, dettype, pp_mode, analysis_type):

        assert pp_mode in PumpProbeAnalysis._pp_mode.keys()
        assert analysis_type in PumpProbeAnalysis._analysis_type.keys()

        self.dettype = dettype.upper()
        assert self.dettype in ["AGIPD", "LPD", "JUNGFRAU"]

        self.pp_mode = PumpProbeAnalysis._pp_mode[pp_mode]
        self.analysis_type = PumpProbeAnalysis._analysis_type[analysis_type]

        self.run = detector_data_collection(proposal, run, dettype)
        self.assembler =ImageAssembler.for_detector(dettype)
        self.on = None
        self.off = None
        self.difference = None

    def process(self, **kwargs):
        if self.analysis_type == AnalysisType.ROI:
            roi = kwargs.get("roi", None)
            background = kwargs.get("bkg", None)
            fom_type = kwargs.get("fom_type", None)

            if roi is None or fom_type not in ["mean", "proj"]:
                print(f"Check roi {roi} and fom_type {fom_type}")
                return
        train_ids = []
        on = []
        off = []
        diff = []
        for tid, data in self.run.trains():

            assembled = self.assembler.assemble_image(data)
            on_image, off_image = self._on_off_data(tid, assembled)

            if on_image is not None and off_image is not None:
                train_ids.append(tid)
                if self.analysis_type == AnalysisType.ROI:
                    x0, x1, y0, y1 = roi
                    signal_on = on_image[..., x0:x1, y0:y1]
                    signal_off = off_image[..., x0:x1, y0:y1]

                    if background is not None:
                        bx0, bx1, by0, by1 = background
                        signal_on -= on_image[..., bx0:bx1, by0:by1]
                        signal_off -= off_image[..., bx0:bx1, by0:by1]

                    if fom_type == 'proj':
                        on_fom = np.nanmean(signal_on, axis=-1)
                        off_fom = np.nanmean(signal_off, axis=-1)

                    else:
                        on_fom = np.nanmean(signal_on, axis=(-1, -2))
                        off_fom = np.nanmean(signal_off, axis=(-1, -2))

                elif self.analysis_type == AnalysisType.AZIMUTHAL:
                    pass

                on.append(on_fom)
                off.append(off_fom)
                diff.append(on_fom - off_fom)

        if trains:
            coords = {'trainId': np.array(train_ids)}
            dims = ['trainId', 'mem_cells'] + \
                   [f'd{i}' for i in range(len(on.shape[1:]))]
            self.on = xr.DataArray(
                data=np.stack(on), dims=dims, coords=coords)
            self.off = xr.DataArray(
                data=np.stack(off), dims=dims, coords=coords)
            self.diff = xr.DataArray(
                data=np.stack(diff), dims=dims, coords=coords)
