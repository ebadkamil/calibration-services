"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
import pytest
import unittest
import xarray as xr

from calibration.processor.roi_intensity import ModuleRoiIntensity
from ...config import config


@pytest.mark.usefixtures("mock_spb_raw_run")
class TestModuleRoiIntensity(unittest.TestCase):

    def setUp(self):
        _, prop, run = self.run_path.split("/")
        config["test_case"] = True

        self.instance = ModuleRoiIntensity(6, prop, run, "AGIPD")
        self.num_trains = len(self.instance.run.train_ids)

        self.dark_run = np.random.rand(80, 512, 128)

    def test_data_collection(self):
        self.assertEqual(len(self.instance.run.instrument_sources), 1)
        self.assertTrue(len(self.instance.control.control_sources) > 0)

    def test_eval_roi_intensity(self):
        _, _= self.instance.eval_module_roi_intensity()
        self.assertTrue(
            isinstance(self.instance.roi_intensity, xr.DataArray))
        self.assertTrue(
            isinstance(self.instance.roi_intensity_ma, np.ndarray))
        self.assertEqual(
            self.instance.roi_intensity.shape, (self.num_trains, 1, 64))
        self.assertEqual(
            self.instance.roi_intensity_ma.shape, (1, 64))

        # Test with multiple ROIs and pulse_ids
        rois = [[0, 128, 0, 128], [0, 64, 64, 128]]
        pulse_ids = "0:20:2"
        _, _= self.instance.eval_module_roi_intensity(
            rois=rois, pulse_ids=pulse_ids, dark_run=self.dark_run)

        self.assertEqual(
            self.instance.roi_intensity.shape, (self.num_trains, 2, 10))
        self.assertEqual(
            self.instance.roi_intensity_ma.shape, (2, 10))

        # dark data does not have enough pulses
        with self.assertRaises(IndexError):
            _, _= self.instance.eval_module_roi_intensity(
                rois=rois, pulse_ids=pulse_ids,
                dark_run=np.random.rand(4, 512, 128))

    def test_scan(self):
        rois = [[0, 128, 0, 128], [0, 64, 64, 128]]
        pulse_ids = "0:20:2"

        _, _= self.instance.eval_module_roi_intensity(
            rois=rois, pulse_ids=pulse_ids)

        fig = self.instance.plot_scan("SPB_XTD9_XGM/DOOCS/MAIN",
                                    "pulseEnergy.photonFlux.value")

        self.assertIsNotNone(fig)
