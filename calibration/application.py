"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import argparse
import h5py

from .processor import EvalHistogram
from .helpers import find_proposal
from .webapp import DashApp


def detector_characterize():
    parser = argparse.ArgumentParser(prog="detectorCharacterize")
    parser.add_argument("detector", help="detector name (case insensitive)",
                        choices=[det.upper() for det in ["AGIPD", "LPD"]],
                        type=lambda s: s.upper())
    parser.add_argument('module', type=int,
                        choices=list(range(16)),
                        help="Module number")
    parser.add_argument("--proposal", help="Proposal number", required=True)
    parser.add_argument("--run", help="Run number", required=True)
    parser.add_argument("--bin_low", help="lower limit energy", required=True)
    parser.add_argument("--bin_high", help="upper limit energy", required=True)
    parser.add_argument("--nbins", help="Number of bins", required=True)
    parser.add_argument("--dark", help="path to file with dark data")

    args = parser.parse_args()
    detector = args.detector
    module = args.module
    proposal = args.proposal
    run = args.run

    low = args.bin_low
    high = args.bin_high
    nbins = args.nbins

    dark_file_path = args.dark

    if dark_file_path is not None:
        try:
            with h5py.File(dark_file_path, "r") as f:
                dark_data = f["entry_1/instrument/module_{module}/data"][:]
        except Exception as ex:
            print(ex)

    run_path = find_proposal(proposal, run)


def run_dashservice():
    app = DashApp()
    app._app.run_server(debug=False)
