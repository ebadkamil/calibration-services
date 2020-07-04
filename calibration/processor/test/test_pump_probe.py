"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil < ebad.kamil@xfel.eu >
Copyright(C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
import pytest
import unittest

from extra_data import by_index
from calibration.processor.pump_probe_analysis import PumpProbeAnalysis
from ...config import config


@pytest.mark.usefixtures("mock_spb_raw_run")
class TestPumpProbeAnalysis(unittest.TestCase):

    def setUp(self):
        config["test_case"] = True
        _, self.prop, self.run = self.run_path.split("/")

    def test_init(self):
        config["test_case"] = True
        with self.assertRaises(AssertionError):
            instance = PumpProbeAnalysis(
                self.prop, self.run, "AGIPD", "even_dd", "roi")

        with self.assertRaises(AssertionError):
            instance = PumpProbeAnalysis(
                self.prop, self.run, "AGIPD", "same_train", "roi")

        with self.assertRaises(AssertionError):
            instance = PumpProbeAnalysis(
                self.prop, self.run, "AGIPD", "same_train", "roi",
                on_pulses="0:10:2", off_pulses="0:20:1")

    def test_even_odd(self):

        instance = PumpProbeAnalysis(
            self.prop, self.run, "AGIPD", "even_odd", "roi")
        # Use only first 10 trains for test
        instance.run = instance.run.select_trains(by_index[0:10])
        instance.control = instance.control.select_trains(by_index[0:10])

        on, off, fom = instance.process(
            roi=[300, 400, 800, 900], fom_type="proj")

        self.assertEqual(on.shape[-1], 100)
        self.assertEqual(off.shape[-1], 100)
        self.assertEqual(on.shape, off.shape)
        self.assertEqual(fom.shape[-1], 1)

    def test_same_train(self):

        instance = PumpProbeAnalysis(
            self.prop, self.run, "AGIPD", "same_train", "roi",
            on_pulses="0:10:2", off_pulses="1:10:2")
        # Use only first 10 trains for test
        instance.run = instance.run.select_trains(by_index[0:10])
        instance.control = instance.control.select_trains(by_index[0:10])

        on, off, fom = instance.process(
            roi=[300, 400, 800, 900], fom_type="proj")

        self.assertEqual(on.shape[-1], 100)
        self.assertEqual(off.shape[-1], 100)
        self.assertEqual(on.shape, off.shape)
        self.assertEqual(fom.shape[-1], 1)

    def test_fom_scan(self):
        instance = PumpProbeAnalysis(
            self.prop, self.run, "AGIPD", "even_odd", "roi")
        # Use only first 10 trains for test
        instance.run = instance.run.select_trains(by_index[0:10])
        instance.control = instance.control.select_trains(by_index[0:10])

        on, off, fom = instance.process(
            roi=[300, 400, 800, 900], fom_type="proj")
        scan, mean, std, fig = instance.fom_scan(
            "SPB_XTD9_XGM/DOOCS/MAIN", "pulseEnergy.photonFlux.value")

        self.assertEqual(mean.shape, std.shape)
        self.assertEqual(len(scan.shape), 1)
        self.assertEqual(mean.shape[0], scan.shape[0])
