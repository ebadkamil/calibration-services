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
from .helpers import find_proposal, parse_ids
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
    parser.add_argument("--pulseids", type=str, help="Pulses '1:20:2'")
    parser.add_argument("--eval_dark", action='store_true',
                        help="To evaluate dark run")
    parser.add_argument("--subtract_dark", action='store_true',
                        help="To subtract dark from a run")
    parser.add_argument("--pixel_hist", action='store_true',
                        help="To evaluate pixel histogram")
    parser.add_argument("--fit", action='store_true',
                        help="To fit histograms")

    args = parser.parse_args()
    detector = args.detector
    module = args.module
    proposal = args.proposal
    run = args.run

    low = args.bin_low
    high = args.bin_high
    nbins = args.nbins

    pixel_hist = args.pixel_hist
    pulse_ids = args.pulseids

    eval_dark = args.eval_dark
    subtract_dark = args.subtract_dark

    fit = args.fit

    if eval_dark:
        counts_file = os.path.join(os.getcwd(), f"dark_module_{module}.h5")
    else:
        counts_file = os.path.join(os.getcwd(), f"data_module_{module}.h5")

    dark_data = None

    if subtract_dark:
        file = os.path.join(os.getcwd(), f"dark_module_{module}.h5")
        with h5py.File(file, "r") as f:
            dark_data = f[f"entry_1/instrument/module_{module}/image"][:]
        print("Shape of dark data, ", dark_data.shape)

    run_path = find_proposal(proposal, run)
    bin_edges = np.linspace(low, high, nbins)

    print('-' * 30)
    print('{:<20s}{:>4s}'.format("Argument", "Value"))
    print('-' * 30)
    for arg, value  in vars(args).items():
        print('{:<20s}{:>4s}'.format(arg, str(value)))

    t0 = time.perf_counter()
    e = EvalHistogram(
        module, run_path, detector, pixel_hist=pixel_hist)
    _ = e.process(
        bin_edges, workers=5, pulse_ids=pulse_ids, dark_run=dark_data)
    e.hist_to_file(counts_file)
    print(f"Time taken for histogram Eval.: {time.perf_counter()-t0}")

    # Fixed initial parameters
    # TODO: Make it configurable
    if fit:
        t0 = time.perf_counter()
        fit_file = os.path.join(os.getcwd(), f"fit_params_{module}.h5")
        params = [100, 70, 50, 10, 10, 10, -25, 25, 70]
        bounds_minuit = [(0, None), (0, None), (0, None),
                         (0, None), (0, None), (0, None),
                         (-50, 0), (0, 50), (40, 100)]

        _ = e.fit_histogram(params, bounds_minuit)
        e.fit_params_to_file(fit_file)

        print(f"Time taken for Fitting: {time.perf_counter()-t0}")


def run_dashservice():
    app = DashApp()
    app._app.run_server(debug=False)
