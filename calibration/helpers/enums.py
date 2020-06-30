"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import IntEnum


class PumpProbeMode(IntEnum):
    EVEN_ODD = 1 # Even train On, ODD train Off
    ODD_EVEN = 2 # Odd train ON, Even train Off
    SAME_TRAIN = 3 # ON OFF in same train for pulse resolved detector


class AnalysisType(IntEnum):
    ROI = 1
    AZIMUTHAL = 2
