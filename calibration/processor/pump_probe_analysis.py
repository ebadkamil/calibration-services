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
    EVEN_ODD = 1 # Even train On, ODD train Off
    ODD_EVEN = 2 # Odd train ON, Even train Off
    SAME_TRAIN = 3 # ON OFF in same train for pulse resolved detector


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

    def __init__(self,
                 proposal,
                 run,
                 dettype,
                 pp_mode,
                 analysis_type,
                 on_pulses=None,
                 off_pulses=None):

        assert pp_mode in PumpProbeAnalysis._pp_mode.keys()
        assert analysis_type in PumpProbeAnalysis._analysis_type.keys()

        self.dettype = dettype.upper()
        assert self.dettype in ["AGIPD", "LPD", "JUNGFRAU"]

        self.pp_mode = PumpProbeAnalysis._pp_mode[pp_mode]

        if self.pp_mode == PumpProbeMode.SAME_TRAIN:
            assert all([on_pulses is not None, off_pulses is not None])
            self.on_pulses = parse_ids(on_pulses)
            self.off_pulses = parse_ids(off_pulses)
            assert self._validate_on_off_pulse_pattern(
                self.on_pulses, self.off_pulses)

        self.analysis_type = PumpProbeAnalysis._analysis_type[analysis_type]

        self.run = detector_data_collection(proposal, run, dettype)
        self.assembler =ImageAssembler.for_detector(dettype)
        self.on = None
        self.off = None
        self.difference = None
        self._prev_on = None

    def process(self, **kwargs):
        if self.analysis_type == AnalysisType.ROI:
            roi = kwargs.get("roi", None)
            background = kwargs.get("bkg", None)
            fom_type = kwargs.get("fom_type", None)

            if roi is None or fom_type not in ["mean", "proj"]:
                print(f"Check roi {roi} and fom_type {fom_type}")
                return

            if roi is not None and background is not None:
                if not self._validate_rois(roi, background):
                    print("Signal and background roi are of different size",
                          f"{signal:{roi} background:{background}}")
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
            dims = ['trainId'] + \
                   [f'd{i}' for i in range(len(on.shape[1:]))]
            self.on = xr.DataArray(
                data=np.stack(on), dims=dims, coords=coords)
            self.off = xr.DataArray(
                data=np.stack(off), dims=dims, coords=coords)
            self.diff = xr.DataArray(
                data=np.stack(diff), dims=dims, coords=coords)

    def _on_off_data(self, tid, image):
        if self.pp_mode = PumpProbeMode.SAME_TRAIN:
            on_image = np.nanmean(image[self.on_pulses, ...], axis=0)
            off_image = np.nanmean(image[self.off_pulses, ...], axis=0)

        if self.pp_mode in [PumpProbeMode.EVEN_ODD, PumpProbeMode.ODD_EVEN]:
            flag = 0 if self.pp_mode == PumpProbeMode.EVEN_ODD else 1

            if tid % 2 == 1 ^ flag:
                self._prev_on = np.nanmean(image, axis=0)

            else:
                on_image = self._prev_on
                self._prev_on = None
                off_image = np.nanmean(image, axis=0)

        return on_image, off_image

    def _validate_on_off_pulse_pattern(self, on_pulses, off_pulses):
        size_check = len(on_pulses) == len(off_pulses)
        return all(
            [size_check] + [pulse not in off_pulses for pulse in on_pulses])

    def _validate_rois(self, roi1, roi2):
        x0, x1, y0, y1 = roi1
        bx0, bx1, by0, by1 = roi2
        return all([x1-x0 == bx1-bx0, y1-y0 == by1-by0])