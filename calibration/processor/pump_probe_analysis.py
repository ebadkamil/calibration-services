"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

from collections import OrderedDict
from enum import IntEnum
import fnmatch
import os.path as osp
import os
import re
from scipy import constants

import numpy as np
import xarray as xr

from karabo_data import DataCollection, by_index

from .assembler import ImageAssembler
from .descriptors import MovingAverage, PyFaiAzimuthalIntegrator
from ..gui.plots import ScatterPlot
from ..helpers import (
    AnalysisType, detector_data_collection, find_proposal, PumpProbeMode,
    slice_curve, timeit)


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
                 off_pulses=None,
                 data='proc'):

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

        self.run_path = find_proposal(proposal, run, data=data)
        self.run = detector_data_collection(
            proposal, run, dettype, data=data)

        self.assembler =ImageAssembler.for_detector(dettype)

        self.on = None
        self.off = None
        self.fom = None
        self._prev_on = None

    def process(self, **kwargs):
        if self.analysis_type == AnalysisType.ROI:
            roi = kwargs.get("roi", None)
            background = kwargs.get("bkg", None)
            fom_type = kwargs.get("fom_type", None)
            auc = kwargs.get("auc", [None, None])

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
        foms = []

        for tid, data in self.run.trains():
            assembled = self.assembler.assemble_image(data)
            if assembled is None:
                continue
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
                        on_fom = np.nanmean(signal_on, axis=-2)
                        # Normalize on spectra with area under curve
                        on_fom /= np.trapz(*slice_curve(
                                on_fom, np.arange(on_fom.shape[-1])))

                        off_fom = np.nanmean(signal_off, axis=-2)
                        # Normalize off spectra with area under curve
                        off_fom /= np.trapz(*slice_curve(
                                off_fom, np.arange(off_fom.shape[-1])))
                    else:
                        on_fom = np.nanmean(signal_on, axis=(-1, -2))
                        off_fom = np.nanmean(signal_off, axis=(-1, -2))

                elif self.analysis_type == AnalysisType.AZIMUTHAL:
                    continue

                diff = np.abs(on_fom - off_fom)
                if diff.shape[-1] == 1:
                    # Just a number in case of mean roi
                    fom = diff
                else:
                    # A curve in case of azimuthal integration or projection
                    fom = np.trapz(*slice_curve(
                                diff, np.arange(diff.shape[-1]), *auc))

                on.append(on_fom)
                off.append(off_fom)
                foms.append(fom)

        if train_ids:
            coords = {'trainId': np.array(train_ids)}
            dims = ['trainId'] + \
                   [f'd{i}' for i in range(len(np.stack(on).shape[1:]))]
            self.on = xr.DataArray(
                data=np.stack(on), dims=dims, coords=coords)
            self.off = xr.DataArray(
                data=np.stack(off), dims=dims, coords=coords)

            dims = ['trainId'] + \
                   [f'd{i}' for i in range(len(np.stack(foms).shape[1:]))]
            self.fom = xr.DataArray(
                data=np.stack(foms), dims=dims, coords=coords)

            return self.on, self.off, self.fom

    def fom_scan(self, src, prop):
        if self.fom is None:
            print("Figure of merit is not available")
            return

        files = [f for f in os.listdir(self.run_path) if f.endswith('.h5')]
        files = [os.path.join(self.run_path, f)
                 for f in fnmatch.filter(files, '*DA*')]

        scan_data = DataCollection.from_paths(files).get_array(src, prop)

        assert len(scan_data.shape) == 1

        align = xr.merge(
            [self.fom.rename('fom'),
             scan_data.rename('scan_data')],
             join='inner')

        # Take mean and std after grouping with scan data
        mean_align = align.groupby('scan_data').mean(dim=['trainId'])
        std_align = align.groupby('scan_data').std(dim=['trainId'])

        # Create ScatterPlot object
        fig = ScatterPlot(title=f'Pump Probe Analysis',
                          xlabel=f"Scan variable ({src}/{prop})",
                          ylabel="FOM",
                          legend='Module',
                          )

        # Set data
        fig.setData(
            mean_align['scan_data'],
            mean_align['fom'].squeeze(axis=-1),
            yerror=std_align['fom'].squeeze(axis=-1)
            )

        return (mean_align['scan_data'].values,
                mean_align['fom'].values,
                std_align['fom'].values,
                fig)

    def _on_off_data(self, tid, image):
        on_image = None
        off_image = None
        if self.pp_mode == PumpProbeMode.SAME_TRAIN:
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
