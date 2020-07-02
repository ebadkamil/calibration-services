"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import pytest
from tempfile import TemporaryDirectory

from extra_data.tests import make_examples


@pytest.fixture(scope='class')
def mock_spb_raw_run(request):
    with TemporaryDirectory() as td:
        make_examples.make_spb_run(td, format_version='1.0')
        request.cls.run_path = td
        yield
