"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import h5py
import ipywidgets as widgets
from ipywidgets import Button
from IPython.display import display
import numpy as np
import plotly.graph_objs as go
import re

from karabo_data import by_index, RunDirectory, stack_detector_data
from karabo_data.geometry2 import AGIPD_1MGeometry

from .logger import out, logger
from ..helpers import parse_le, parse_ids, pulse_filter
from ..processor import DataProcessing, DataModel, eval_statistics, gauss_fit


class SimpleImageViewer:
    '''Control panel'''

    def __init__(self, config=None):
        self._cntrl = widgets.Tab()
        self.config = config
        self.futures = {}
        self.data_model = {}
        self.filtered = None
        self._cntrl = self._initDarkRun()
        self.geom = AGIPD_1MGeometry.from_quad_positions(quad_pos=[
            (-525, 625),
            (-550, -10),
            (520, -160),
            (542.5, 475),
        ])
        self.config = config
        self.dark_data = {}
        self.out_array = None
        self.assembled = None
        self.run = None

    def _initDarkRun(self):

        item_layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            justify_content='space-between')

        self._run_folder = widgets.Text(
            value=self.config['run_folder'])

        self._load_run = Button(description='Load Run')
        self._load_run.button_style = 'success'
        self._load_run.on_click(self._on_load_run)

        self._dark_data_path = widgets.Text(
            value=self.config['dark_data'])

        self._train_ids = widgets.BoundedIntText(
            description='Train Ids:',
            continuous_update=False,
            disabled=True)

        self._pulse_indices = widgets.BoundedIntText(
            description='Pulse Ids:',
            min=0,
            max=2700,
            step=1,
            value=0,
            continuous_update=False)

        self._bins = widgets.IntText(
            description='Bins:',
            value=100,
            continuous_update=False)

        self._train_ids.observe(self.onTrainIdChange, names='value')
        self._pulse_indices.observe(self.onVisulizationParamChange, names='value')
        self._bins.observe(self.onVisulizationParamChange, names='value')

        items_params = [
            widgets.Box([self._train_ids, self._pulse_indices, self._bins],
                layout=item_layout),
        ]

        params = widgets.Box(items_params, layout=widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='stretch',
            width='40%'
        ))

        ctrl = widgets.VBox(
            [widgets.Label(value='Run Folder:'),
             self._run_folder,
             widgets.Label(value='Dark data path:'),
             self._dark_data_path,
             self._load_run],
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',
                align_items='stretch',
                width='50%',
                justify_content='space-between'))

        self._image_widget = go.FigureWidget(
            data=go.Heatmap(showscale=False))
        self._hist_widget = go.FigureWidget(data=go.Bar())

        self._plot_widgets = widgets.HBox(
            [self._image_widget,
             self._hist_widget
             ])
        self._image_widget.layout.update(
            margin=dict(l=0, b=40, t=50), width=450)
        self._hist_widget.layout.update(
            margin=dict(r=0, l=10, b=40, t=50), width=450)

        self._ctrl_widget = widgets.HBox(
            [ctrl,
             params],
            layout=widgets.Layout(
                display='flex',
                flex_flow='row',
                align_items='stretch',
                width='100%',
                justify_content='space-between'))

        return widgets.VBox(
            [self._ctrl_widget,
             self._plot_widgets],
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',
                align_items='stretch',
                width='100%',
                justify_content='space-between'))

    def _on_load_run(self, e=None):
        self._load_run.disabled = True
        run_path = self._run_folder.value
        devices = [("*/DET/*CH0:xtdf", "image.data")]
        if run_path:
            try:
                self.run = RunDirectory(run_path).select(devices)
            except Exception as ex:
                return
        self._train_ids.min = 0
        self._train_ids.value = 0
        self._train_ids.max = len(self.run.train_ids) - 2
        self._train_ids.step = 1
        self._load_run.disabled = False

        self.tid, self.train_data = self.run.train_from_index(
            self._train_ids.value)

        dark_path = self._dark_data_path.value
        if dark_path:
            try:
                def iterate(name, node):
                    if isinstance(node, h5py.Dataset):
                        m = re.search("(.+)module_(.+)/data", name)
                        if m is not None:
                            self.dark_data[m.group(2)] = node[:]

                with h5py.File(dark_path, 'r') as f:
                    f.visititems(iterate)
            except Exception as ex:
                print(ex)

        self._assemble_image()

    def _assemble_image(self):

        def _corrections(source):
            pattern = "(.+)/DET/(.+)CH0:xtdf"
            modno = int((re.match(pattern, source)).group(2).strip())

            image = self.train_data[source]["image.data"][:, 0, ...]
            image = image.astype(np.float32)

            if self.dark_data and image.shape[0] != 0:
                image -= self.dark_data[str(modno)][0:image.shape[0], ...]

            self.train_data[source]["image.data"] = image

        with ThreadPoolExecutor(
                max_workers=len(self.train_data.keys())) as executor:
            for source in self.train_data.keys():
                executor.submit(_corrections, source)
        # assemble image
        try:
            stacked_data = stack_detector_data(self.train_data, "image.data")
        except (ValueError, IndexError, KeyError) as e:
            self._train_ids.disabled = False
            return

        n_images = (stacked_data.shape[0], )
        if stacked_data.shape[0] == 0:
            self._train_ids.disabled = False
            return

        image_dtype = stacked_data.dtype

        if self.out_array is None:
            self.out_array = self.geom.output_array_for_position_fast(
                extra_shape=n_images, dtype=image_dtype)

        self.assembled, centre = self.geom.position_all_modules(
            stacked_data, out=self.out_array)

        self.onVisulizationParamChange(0)
        self._train_ids.disabled = False

    def onVisulizationParamChange(self, value):
        pulse = self._pulse_indices.value
        nbins = self._bins.value if self._bins.value > 2 else 100
        if self.assembled is not None:
            try:
                img_to_plot = self.assembled[pulse]
            except Exception as ex:
                return

        self._image_widget.data[0].z = img_to_plot[::2, ::2]

        img_to_plot[np.isnan(img_to_plot)] = 0.0
        counts, bins = np.histogram(img_to_plot.ravel(), bins=nbins)
        self._hist_widget.data[0].x = (bins[1:] + bins[:-1]) / 2.0
        self._hist_widget.data[0].y = counts

    def onTrainIdChange(self, value):
        self._train_ids.disabled = True
        if value['new'] > len(self.run.train_ids) - 1:
            self._train_ids.disabled = False
            return
        if self.run is not None:
            self.tid, self.train_data = self.run.train_from_index(
                self._train_ids.value)
            self._assemble_image()

    def control_panel(self):
        display(self._cntrl)
