"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import argparse
import os
import time

import h5py
import numpy as np

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
    parser.add_argument("--bin_low", type=float, required=True,
                        help="lower limit energy")
    parser.add_argument("--bin_high", type=float,
                        help="upper limit energy", required=True)
    parser.add_argument("--nbins", type=int,
                        help="Number of bins", required=True)
    parser.add_argument("--pulseids", help="Pulses '1:20:2'")
    parser.add_argument("--dark", help="path to file with dark data")
    parser.add_argument("--pixel_hist", action='store_true',
                        help="To evaluate pixel histogram")

    args = parser.parse_args()
    detector = args.detector
    module = args.module
    proposal = args.proposal
    run = args.run

    low = args.bin_low
    high = args.bin_high
    nbins = args.nbins

    dark_file_path = args.dark
    pixel_hist = args.pixel_hist
    pulse_ids = args.pulseids


    if dark_file_path is not None:
        # key should be mean_image
        with h5py.File(dark_file_path, "r") as f:
            dark_data = f[f"entry_1/instrument/module_{module}/data"][:]
        print("Shape of dark data, ", dark_data.shape)

    run_path = find_proposal(proposal, run)
    bin_edges = np.linspace(low, high, nbins)
    counts_file = os.path.join(os.getcwd(), f"data_module_{module}.h5")

    t0 = time.perf_counter()
    e = EvalHistogram(
        module, run_path, detector, pixel_hist=pixel_hist)
    e.process(bin_edges, workers=5, pulse_ids=pulse_ids, dark_run=dark_data)
    e.hist_to_file(counts_file)
    print(f"Time taken for histogram Eval.: {time.perf_counter()-t0}")


def run_dashservice():
    app = DashApp()
    app._app.run_server(debug=False)
