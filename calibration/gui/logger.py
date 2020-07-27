"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

import logging
import ipywidgets as widgets
import sys


out = widgets.Output(
    layout=widgets.Layout(width='100%', height='160px', border='1px solid'))

class LoggerWidget(logging.Handler):
    """ Class for logging data """

    def __init__(self, *args, **kwargs):
         # Initialize the Handler
         super().__init__(*args, **kwargs)

    def emit(self, record):
        """ Overload of logging.Handler method """

        record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': record+'\n'
        }
        out.outputs = (new_output, ) + out.outputs

logger = logging.getLogger(__name__)

handler = LoggerWidget()
handler.setFormatter(
    logging.Formatter('%(asctime)s  - [%(levelname)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

