"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

from concurrent.futures import ThreadPoolExecutor

import h5py
import ipywidgets as widgets
from ipywidgets import Button
from IPython.display import display
import numpy as np
import plotly.graph_objs as go
import re

from karabo_data import RunDirectory, stack_detector_data
from karabo_data.geometry2 import AGIPD_1MGeometry, LPD_1MGeometry

from .logger import out


class SimpleImageViewer:
    '''Control panel'''

    def __init__(self, dettype, config=None):
        self.config = config
        self._cntrl = self._initDarkRun()

        self.dettype = dettype
        assert self.dettype in ["AGIPD", "LPD"]
        if self.dettype == 'AGIPD':
            self.geom = AGIPD_1MGeometry.from_quad_positions(
                quad_pos=[
                    (-525, 625),
                    (-550, -10),
                    (520, -160),
                    (542.5, 475), ])
        else:
            self.geom = LPD_1MGeometry.from_quad_positions(
                quad_pos=[
                    [11.4, 299],
                    [-11.5, 8],
                    [254.5, -16],
                    [278.5, 275]],)

        self.dark_data = {}
        self.out_array = None
        self.train_data = None
        self.assembled = None
        self.run = None

        out.clear_output()

    def _initDarkRun(self):

        item_layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            justify_content='space-between')

        self._run_folder = widgets.Text(
            value=self.config['run_folder'],
            description='Run Folder:',)

        self._load_run = Button(description='Load Run')
        self._load_run.button_style = 'success'
        self._load_run.on_click(self._on_load_run)

        self._dark_data_path = widgets.Text(
            value=self.config['dark_data'],
            description='Dark data:')

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
            disabled=True,
            continuous_update=False)

        self._bins = widgets.IntText(
            description='Bins:',
            value=100,
            disabled=True,
            continuous_update=False,)

        self._threshold_mask = widgets.FloatRangeSlider(
            value=[-10000, 10000],
            min=-10000,
            max=10000,
            step=10.0,
            description='Mask:',
            disabled=True,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f')

        self._train_ids.observe(self.onTrainIdChange, names='value')
        self._pulse_indices.observe(
            self.onVisulizationParamChange, names='value')
        self._bins.observe(self.onVisulizationParamChange, names='value')
        self._threshold_mask.observe(
            self.onVisulizationParamChange, names='value')

        items_params = [
            widgets.Box([self._train_ids,
                         self._pulse_indices,
                         self._bins,
                         self._threshold_mask],
                        layout=item_layout),
        ]

        params = widgets.Box(items_params, layout=widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='stretch',
            width='40%'
        ))

        ctrl = widgets.VBox(
            [self._run_folder,
             self._dark_data_path,
             self._load_run],
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',
                align_items='stretch',
                width='50%',
                justify_content='space-between'))

        self._image_widget = go.FigureWidget(
            data=go.Heatmap())
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

    @out.capture()
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
                print("Dark offset was not applied")
                self.dark_data = {}

        self._assemble_image()

    @out.capture()
    def _assemble_image(self):
        if self.train_data is None:
            return

        self.assembled = None

        def _corrections(source):
            pattern = "(.+)/DET/(.+)CH0:xtdf"
            modno = int((re.match(pattern, source)).group(2).strip())
            if self.dettype == "AGIPD":
                image = self.train_data[source]["image.data"][:, 0, ...]
            else:
                image = np.squeeze(
                    self.train_data[source]["image.data"], axis=1)
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
            print(e)

        n_images = (stacked_data.shape[0], )
        if stacked_data.shape[0] == 0:
            self._train_ids.disabled = False
            print("Number of pulses for train are 0")

        image_dtype = stacked_data.dtype

        if self.out_array is None:
            self.out_array = self.geom.output_array_for_position_fast(
                extra_shape=n_images, dtype=image_dtype)

        self.assembled, centre = self.geom.position_all_modules(
            stacked_data, out=self.out_array)

        self.onVisulizationParamChange(0)

    @out.capture()
    def onVisulizationParamChange(self, value):
        description = None
        if value != 0:
            description = value['owner'].description
        self.widgetsOnParamsChange()

        pulse = self._pulse_indices.value
        nbins = self._bins.value if self._bins.value > 2 else 100
        start, stop = self._threshold_mask.value

        if self.assembled is not None:
            try:
                img_to_plot = np.copy(self.assembled[pulse])
                img_to_plot[(img_to_plot < start) | (
                    img_to_plot > stop)] = np.nan
            except Exception as ex:
                print(ex)
                self.widgetsOnParamsChanged()
                return

        if value == 0 or description != "Bins:":
            self._image_widget.data[0].z = img_to_plot

        counts, bins = np.histogram(
            img_to_plot[~np.isnan(img_to_plot)].ravel(), bins=nbins)
        self._hist_widget.data[0].x = (bins[1:] + bins[:-1]) / 2.0
        self._hist_widget.data[0].y = counts
        self.widgetsOnParamsChanged()

    def onTrainIdChange(self, value):
        self.widgetsOnParamsChange()
        self.train_data = None

        if value['new'] > len(self.run.train_ids) - 1:
            self._train_ids.disabled = False
            print("Train Index out of range")
            return
        if self.run is not None:
            self.tid, self.train_data = self.run.train_from_index(
                self._train_ids.value)
            self._assemble_image()

    def widgetsOnParamsChange(self):
        self._train_ids.disabled = True
        self._pulse_indices.disabled = True
        self._bins.disabled = True
        self._threshold_mask.disabled = True

    def widgetsOnParamsChanged(self):
        self._train_ids.disabled = False
        self._pulse_indices.disabled = False
        self._bins.disabled = False
        self._threshold_mask.disabled = False

    def control_panel(self):
        display(self._cntrl, out)
