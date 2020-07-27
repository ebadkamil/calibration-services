"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
import pytest
import unittest

from extra_data import by_index
from calibration.processor.azimuthal_integration import AzimuthalIntegration
from ...config import config


@pytest.mark.usefixtures("mock_spb_raw_run")
class TestAzimuthalIntegration(unittest.TestCase):

    def setUp(self):
        _, prop, run = self.run_path.split("/")
        config["test_case"] = True
        ai_config = dict(
            energy=9.3,
            pixel_size=0.5e-3,
            centrex=580,
            centrey=620,
            distance=0.2,
            intg_rng=[0.2, 5],
            intg_method='BBox',
            intg_pts=512,)

        self.instance = AzimuthalIntegration(
            prop, run, "AGIPD", ai_config, data='raw', window=10)

        self.num_trains = len(self.instance.run.train_ids)

    def test_integrate(self):
        _, _, _= self.instance.integrate(train_index=by_index[0:10])

        self.assertEqual(self.instance.momentums.shape, (10, 512))
        self.assertEqual(self.instance.intensities.shape, (10, 64, 512))
        self.assertEqual(self.instance.intensities_ma.shape, (64, 512))

        _, _, _= self.instance.integrate(pulse_ids="0:30:2",
                                         train_index=by_index[0:10])
        self.assertEqual(self.instance.intensities.shape, (10, 15, 512))
        self.assertEqual(self.instance.intensities_ma.shape, (15, 512))
