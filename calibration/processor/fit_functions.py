"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np


def gaussian(x, *params):
    num_gaussians = int(len(params) / 3)
    A = params[:num_gaussians]
    w = params[num_gaussians:2*num_gaussians]
    c = params[2*num_gaussians:3*num_gaussians]
    y = sum(
        [A[i]*np.exp(-(x-c[i])**2./(w[i])) for i in range(num_gaussians)])
    return y


def least_squares_np(xdata, ydata,  params):
    y = gaussian(xdata, *params)
    return np.sum((ydata - y) ** 2)