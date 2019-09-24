import argparse
from math import ceil
from layout import get_layout
from queue import Queue
import queue
from core import ProcessedData, DataProcessorWorker
import dash
import dash_html_components as html
import dash_core_components as dcc

import numpy as np
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output, State
from utils import get_virtual_memory
from core.config import config


class DashApp:

    def __init__(self, detector):
        app = dash.Dash("Calibration Analysis")
        app.config['suppress_callback_exceptions'] = True

        config["DETECTOR"] = detector
        self._config = config[detector]
        self._app = app

        self._data = ProcessedData(1000)
        self._proc_queue = Queue(maxsize=1)

        self.processor = DataProcessorWorker(self._proc_queue)

        self.setLayout()
        self.register_callbacks()

    def setLayout(self):
        self._app.layout = get_layout(config["TIME_OUT"], self._config)

    def register_callbacks(self):
        """Register callbacks"""

        @self._app.callback(
            [Output('virtual_memory', 'value'),
             Output('virtual_memory', 'max'),
             Output('swap_memory', 'value'),
             Output('swap_memory', 'max')],
            [Input('psutil_component', 'n_intervals')])
        def update_memory_info(n):
            try:
                virtual, swap = get_virtual_memory()
            except Exception:
                raise dash.exceptions.PreventUpdate
            return ((virtual.used/1024**3), ceil((virtual.total/1024**3)),
                    (swap.used/1024**3), ceil((swap.total/1024**3)))

        @self._app.callback(Output('mean-image', 'figure'),
                            [Input('color-scale', 'value'),
                             Input('train-id', 'value')])
        def update_image_figure(color_scale, tid):
            if self._data.tid != int(tid) or self._data.image is None:
                raise dash.exceptions.PreventUpdate

            traces = [go.Heatmap(
                z=self._data.image[::3, ::3], colorscale=color_scale)]
            figure = {
                'data': traces,
                'layout': go.Layout(
                    margin={'l': 40, 'b': 40, 't': 40, 'r': 10},
                )
            }

            return figure

        @self._app.callback(Output('histogram', 'figure'),
                            [Input('train-id', 'value')])
        def update_histogram_figure(tid):
            if self._data.tid != int(tid) or self._data.image is None:
                raise dash.exceptions.PreventUpdate
            hist, bins = np.histogram(self._data.image.ravel(), bins=10)
            bin_center = (bins[1:] + bins[:-1])/2.0
            traces = [{'x': bin_center, 'y': hist,
                       'type': 'bar'}]
            figure = {
                'data': traces,
                'layout': go.Layout(
                    margin={'l': 40, 'b': 40, 't': 40, 'r': 10},
                )
            }

            return figure

        @self._app.callback(
            Output('load-info', 'children'),
            [Input('load', 'on')],
            [State('run-folder', 'value')])
        def load(state, folder):
            info = ""

            return [info]

    def _update(self):
        try:
            self._data = self._proc_queue.get_nowait()
        except queue.Empty:
            self._data = None

    def process(self):
        self.processor.daemon = True
        self.processor.start()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(prog="dashapp")
    ap.add_argument("detector", help="detector name (case insensitive)",
                    choices=[det.upper()
                             for det in ["jungfrau", "LPD", "AGIPD"]],
                    type=lambda s: s.upper())
    args = ap.parse_args()

    detector = args.detector
    if detector == 'JUNGFRAU':
        detector = 'JungFrau'

    app = DashApp(detector)
    # app.process()

    app._app.run_server(debug=False)
