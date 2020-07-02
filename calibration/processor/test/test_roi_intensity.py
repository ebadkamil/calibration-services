"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
import pytest
from tempfile import TemporaryDirectory
import unittest
import xarray as xr

from extra_data.tests import make_examples

from calibration.processor.roi_intensity import ModuleRoiIntensity

from ...config import config


@pytest.fixture(scope='class')
def mock_spb_raw_run(request):
    with TemporaryDirectory() as td:
        make_examples.make_spb_run(td, format_version='1.0')
        request.cls.run_path = td
        yield


@pytest.mark.usefixtures("mock_spb_raw_run")
class TestModuleRoiIntensity(unittest.TestCase):

    def setUp(self):
        _, prop, run = self.run_path.split("/")
        config["test_case"] = True

        self.instance = ModuleRoiIntensity(6, prop, run, "AGIPD")
        self.num_trains = len(self.instance.run.train_ids)

    def test_data_collection(self):
        self.assertEqual(len(self.instance.run.instrument_sources), 1)
        self.assertTrue(len(self.instance.control.control_sources) > 0)

    def test_eval_roi_intensity(self):
        rois = [[0, 128, 0, 128], [0, 64, 64, 128]]
        pulse_ids = "0:20:2"
        _, _= self.instance.eval_module_roi_intensity(
            rois=rois, pulse_ids=pulse_ids)

        self.assertTrue(
            isinstance(self.instance.roi_intensity, xr.DataArray))
        self.assertTrue(
            isinstance(self.instance.roi_intensity_ma, np.ndarray))
        self.assertEqual(
            self.instance.roi_intensity.shape, (self.num_trains, 2, 10))
        self.assertEqual(
            self.instance.roi_intensity_ma.shape, (2, 10))

    def test_scan(self):
        rois = [[0, 128, 0, 128], [0, 64, 64, 128]]
        pulse_ids = "0:20:2"

        _, _= self.instance.eval_module_roi_intensity(
            rois=rois, pulse_ids=pulse_ids)

        fig = self.instance.plot_scan("SPB_XTD9_XGM/DOOCS/MAIN",
                                    "pulseEnergy.photonFlux.value")

        self.assertIsNotNone(fig)

