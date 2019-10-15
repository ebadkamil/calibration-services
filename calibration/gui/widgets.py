"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import ipywidgets as widgets
from ipywidgets import Button
from IPython.display import display
import numpy as np
import plotly.graph_objs as go

from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

from karabo_data import by_index

from .logger import out, logger
from ..helpers import parse_le, parse_ids, pulse_filter
from ..processor import DataProcessing, DataModel, eval_statistics, gauss_fit


class Display:
    '''Control panel'''

    def __init__(self, config=None):
        self._output = widgets.Output()
        self._cntrl = widgets.Tab()
        self.config = config
        self.futures = {}
        self.data_model = {}
        self.filtered = None
        out.clear_output()
        self._initUI()

    def _initUI(self):

        maps = ['Blackbody', 'Reds', 'Viridis', 'Plasma']
        self._cmaps_list = widgets.Dropdown(
            options=maps,
            value='Blackbody',
            description="Cmap:")

        self._module_dd = widgets.Dropdown(
            options='-',
            value='-',
            description="Module:")

        self._memory_sl = widgets.IntSlider(
            value=0,
            min=0,
            max=250,
            step=1,
            orientation='horizontal',
            readout_format='d',
            description="Pulses:",
            continuous_update=False)

        items_vis_params = [
            self._cmaps_list,
            self._module_dd,
            self._memory_sl
        ]

        self._visualization_widgets = widgets.HBox(
            items_vis_params,
            layout=widgets.Layout(
                display='flex',
                flex_flow='row',
                align_items='stretch',
                width='100%'))

        self._memory_sl.observe(
            self.onVisulizationParamChange, names='value')
        self._module_dd.observe(
            self.onVisulizationParamChange, names='value')
        self._cmaps_list.observe(
            self.onVisulizationParamChange, names='value')

        general_set_up = self._initGeneralSetUp()
        dark_run = self._initDarkRun()
        data_processing = self._initDataProcessing()
        cntrl_items = []
        cntrl_items.append(general_set_up)
        cntrl_items.append(dark_run)
        cntrl_items.append(data_processing)

        self._cntrl.children = cntrl_items
        self._cntrl.set_title(0, "General Setup")
        self._cntrl.set_title(1, "Dark Run")
        self._cntrl.set_title(2, "Data Visualization")

    def _initGeneralSetUp(self):
        item_layout = widgets.Layout(
            display='flex',
            flex_flow='row',
            justify_content='space-between')

        self._module_numbers = widgets.Text(readout=str,
                                            continuous_update=False)
        self._pulse_indices = widgets.Text(value="1:250:2")
        self._roi_x = widgets.Text(value=":")
        self._roi_y = widgets.Text(value=":")

        items = [
            widgets.Box([widgets.Label(value='Module numbers:'),
                         self._module_numbers], layout=item_layout),
            widgets.Box([widgets.Label(value='Pulse indices:'),
                         self._pulse_indices], layout=item_layout),
            widgets.Box([widgets.Label(value='ROI x:'),
                         self._roi_x], layout=item_layout),
            widgets.Box([widgets.Label(value='ROI y:'),
                         self._roi_y], layout=item_layout),
        ]

        general_params = widgets.Box(items, layout=widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='stretch',
            width='50%'
        ))

        self._module_numbers.observe(self.onModuleNumbersChange, 'value')
        self._module_numbers.value = "15, 14"  # Set to emit signal
        return general_params

    def onModuleNumbersChange(self, value):
        options_old = set(self._module_dd.options)
        try:
            options = parse_ids(value['new'])
            options_new = set(options)
            for option in options_new:
                self.data_model[option] = DataModel()
            for old in options_old.difference(options_new):
                if old != '-':
                    del self.data_model[old]
            self._module_dd.options = options
        except ValueError:
            pass

    def _initDarkRun(self):

        item_layout = widgets.Layout(
            display='flex',
            flex_flow='row',
            justify_content='space-between')

        self._dark_run_folder = widgets.Text(
            value=self.config['dark_run_folder'])

        self._process_dark = Button(description='Process Dark')
        self._process_dark.on_click(self._on_process_dark)
        self._process_dark.button_style = 'success'

        self._train_ids = widgets.Text(value=":")

        items_dark_params = [
            widgets.Box([widgets.Label(value='Train ids:'),
                         self._train_ids], layout=item_layout),

        ]

        dark_params = widgets.Box(items_dark_params, layout=widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='stretch',
            width='40%'
        ))

        dark_ctrl = widgets.VBox(
            [widgets.Label(value='Dark Run Folder:'),
             self._dark_run_folder,
             self._process_dark],
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',
                align_items='stretch',
                width='50%',
                justify_content='space-between'))

        self._dark_image_widget = go.FigureWidget(
            data=go.Heatmap(showscale=False))
        self._dark_hist_widget = go.FigureWidget(data=go.Bar())

        self._plot_widgets = widgets.HBox(
            [self._dark_image_widget,
             self._dark_hist_widget
             ])
        self._dark_image_widget.layout.update(
            margin=dict(l=0, b=40, t=50), width=450)
        self._dark_hist_widget.layout.update(
            margin=dict(r=0, l=10, b=40, t=50), width=450)

        self._ctrl_widget = widgets.HBox(
            [dark_ctrl,
             dark_params],
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

    def onVisulizationParamChange(self, value):
        modno = self._module_dd.value
        if self._cntrl.selected_index == 1:
            images = self.data_model[modno].dark_data.image
            centers = self.data_model[modno].dark_data.st.bin_centers
            counts = self.data_model[modno].dark_data.st.bin_counts
            if not all([images is not None,
                        centers is not None,
                        counts is not None]):
                return

            else:
                pid = self._memory_sl.value
                shape = images.shape

                if pid > shape[0] - 1:
                    return

                self._dark_image_widget.data[0].z = images[pid, 0, ...]

                self._dark_image_widget.data[0].colorscale = \
                    self._cmaps_list.value

                self._dark_hist_widget.data[0].x = centers[pid]
                self._dark_hist_widget.data[0].y = counts[pid]

        elif self._cntrl.selected_index == 2:
            images = self.data_model[modno].proc_data.image
            centers = self.data_model[modno].proc_data.st.bin_centers
            counts = self.data_model[modno].proc_data.st.bin_counts
            if not all([images is not None,
                        centers is not None,
                        counts is not None]):
                return
            else:
                pid = self._memory_sl.value
                shape = images.shape

                if pid > shape[0] - 1:
                    return

                self._proc_hist_widget.data[1].x = []
                self._proc_hist_widget.data[1].y = []

                self._proc_image_widget.data[0].z = np.mean(
                    images[pid, :, 0, ...], axis=0)

                self._proc_image_widget.data[0].colorscale = \
                    self._cmaps_list.value

                self._proc_hist_widget.data[0].x = centers[pid]
                self._proc_hist_widget.data[0].y = counts[pid]

                self.filtered = gaussian_filter(counts[pid], 1.5)
                self.peaks, _ = find_peaks(
                    self.filtered,
                    height=self._peak_threshold_sl.value,
                    distance=self._peak_distance_sl.value)

                self._proc_hist_widget.data[2].x = centers[pid][self.peaks]
                self._proc_hist_widget.data[2].y = self.filtered[self.peaks]

                self._proc_fit_params_widget.data[0].cells.values = \
                    [centers[pid][self.peaks], self.filtered[self.peaks], []]
        else:
            pass

    def _initDataProcessing(self):

        item_layout = widgets.Layout(
            display='flex',
            flex_flow='row',
            justify_content='space-between')

        self._run_folder = widgets.Text(value=self.config['run_folder'])

        self._process_run = Button(description='Load Run')
        self._process_run.on_click(self._on_process_run)
        self._process_run.button_style = 'success'

        self._proc_train_ids = widgets.Text(value="250:280")
        self._subtract_dark_cb = widgets.Checkbox(value=False, disabled=True)
        self._peak_threshold_sl = widgets.FloatSlider(
            value=0,
            min=0,
            max=100000,
            step=1.0,
            orientation='horizontal',
            readout_format='0.2f',
            continuous_update=False)
        self._peak_distance_sl = widgets.FloatSlider(
            value=0,
            min=1.0,
            max=1000,
            step=1.0,
            orientation='horizontal',
            readout_format='0.2f',
            continuous_update=False)

        self._fitting_bt = Button(description='Fit Histogram', disabled=True)
        self._fitting_bt.on_click(self._on_fitting)

        self._peak_threshold_sl.observe(
            self.onVisulizationParamChange, 'value')
        self._peak_distance_sl.observe(
            self.onVisulizationParamChange, 'value')
        proc_items = [
            widgets.Box(
                [widgets.Label(value='Train ids:'),
                 self._proc_train_ids],
                layout=item_layout),
            widgets.Box(
                [widgets.Label(value='Subtract Dark:'),
                 self._subtract_dark_cb],
                layout=item_layout),
            widgets.Box(
                [widgets.Label(value='Peak Thresh.:'),
                 self._peak_threshold_sl],
                layout=item_layout),
            widgets.Box(
                [widgets.Label(value='Peak Distance.:'),
                 self._peak_distance_sl],
                layout=item_layout)
        ]

        proc_params = widgets.Box(
            proc_items,
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',
                align_items='stretch',
                width='45%'))

        proc_ctrl = widgets.VBox(
            [widgets.Label(value='Run Folder:'),
             self._run_folder,
             widgets.HBox([self._process_run, self._fitting_bt])],
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',
                align_items='stretch',
                width='50%',
                justify_content='space-between'))

        self._proc_image_widget = go.FigureWidget(
            data=go.Heatmap(showscale=False))

        trace = [go.Bar(), go.Scatter(mode='lines'),
                 go.Scatter(mode='markers')]
        self._proc_hist_widget = go.FigureWidget(data=trace)

        self._proc_image_widget.layout.update(
            margin=dict(l=0, b=40, t=50), width=450)
        self._proc_hist_widget.layout.update(
            margin=dict(r=0, l=10, b=40, t=50), width=450)

        trace = [go.Table(
            header=dict(values=["Positions", "Amplitudes", "Width"],
                        fill_color='paleturquoise',
                        align='center'),
            cells=dict(
                fill_color='lavender',
                align='left'))]

        self._proc_fit_params_widget = go.FigureWidget(data=trace)
        self._proc_fit_params_widget.layout.update(
            margin=dict(l=0, b=0, t=50),
            width=900,
            height=300,
            title="Peaks Info.",
            )
        self._proc_plot_widgets = widgets.VBox(
            [widgets.HBox(
                [self._proc_image_widget,
                 self._proc_hist_widget]),
            self._proc_fit_params_widget
            ])

        self._proc_ctrl_widget = widgets.HBox(
            [proc_ctrl,
             proc_params],
            layout=widgets.Layout(
                display='flex',
                flex_flow='row',
                align_items='stretch',
                width='100%',
                justify_content='space-between'))

        return widgets.VBox(
            [self._proc_ctrl_widget,
             self._proc_plot_widgets])

    def _on_process_dark(self, e=None):
        path=self._dark_run_folder.value
        pulse_ids=str(self._pulse_indices.value)

        train_indices=parse_le(str(self._train_ids.value))

        if train_indices == [-1]:
            train_index=None
        else:
            start, stop=train_indices
            train_index=by_index[start:stop]

        roi_x=parse_le(str(self._roi_x.value))
        roi_y=parse_le(str(self._roi_y.value))

        if roi_x == [-1] and roi_y == [-1]:
            rois=None
        elif roi_x == [-1] and roi_y != [-1]:
            start_y, stop_y=roi_y
            rois=by_index[..., :, start_y:stop_y]
        elif roi_x != [-1] and roi_y == [-1]:
            start_x, stop_x=roi_x
            rois=by_index[..., start_x:stop_x, :]
        else:
            start_x, stop_x=roi_x
            start_y, stop_y=roi_y
            rois=by_index[..., start_x:stop_x, start_y:stop_y]

        eval_dark_average=partial(
            DataProcessing,
            path=path,
            pulse_ids=pulse_ids,
            train_index=train_index,
            rois=rois,
            operation=partial(np.mean, axis=0))

        module_numbers = parse_ids(self._module_numbers.value)
        executor = ProcessPoolExecutor(max_workers=len(module_numbers))

        futures = {}
        for mod_no in module_numbers:
            futures[mod_no] = executor.submit(eval_dark_average, mod_no)
            futures[mod_no].arg = mod_no
            futures[mod_no].add_done_callback(self.onProcessDarkDone)

        self._process_dark.disabled=True
        self._subtract_dark_cb.disabled = True
        self._subtract_dark_cb.value = False

    @out.capture()
    def onProcessDarkDone(self, future):
        if future.cancelled():
            print('{}: canceled'.format(future.arg))
        elif future.done():
            error = future.exception()
            if error:
                print('{}: error returned: {}'.format(
                    future.arg, error))
            else:
                self.data_model[future.arg].dark_data.image=future.result()
                images = self.data_model[future.arg].dark_data.image
                centers_pr=[]
                counts_pr=[]

                def _eval_stat(pulse):
                    centers, counts = eval_statistics(
                        images[pulse, 0, ...], bins=1000)
                    return centers, counts

                with ThreadPoolExecutor(max_workers=10) as executor:
                    for ret in executor.map(_eval_stat, range(images.shape[0])):
                        centers_pr.append(ret[0])
                        counts_pr.append(ret[1])

                self.data_model[future.arg].dark_data.st.bin_centers = np.stack(
                    centers_pr)
                self.data_model[future.arg].dark_data.st.bin_counts = np.stack(
                    counts_pr)

                if future.arg == self._module_dd.value:
                    pid = self._memory_sl.value

                    self._dark_image_widget.data[0].z = images[pid, 0, ...]
                    self._dark_hist_widget.data[0].x = \
                        self.data_model[future.arg].dark_data.st.bin_centers[pid]
                    self._dark_hist_widget.data[0].y = \
                        self.data_model[future.arg].dark_data.st.bin_counts[pid]

                self._subtract_dark_cb.disabled = False

        self._process_dark.disabled = False

    def _on_process_run(self, e=None):
        path = self._run_folder.value
        pulse_ids = str(self._pulse_indices.value)

        proc_train_indices = parse_le(
            str(self._proc_train_ids.value))

        if proc_train_indices == [-1]:
            train_index = None
        else:
            start, stop = proc_train_indices
            train_index = by_index[start:stop]

        roi_x = parse_le(str(self._roi_x.value))
        roi_y = parse_le(str(self._roi_y.value))

        if roi_x == [-1] and roi_y == [-1]:
            rois = None
        elif roi_x == [-1] and roi_y != [-1]:
            start_y, stop_y=roi_y
            rois = by_index[..., :, start_y:stop_y]
        elif roi_x != [-1] and roi_y == [-1]:
            start_x, stop_x = roi_x
            rois = by_index[..., start_x:stop_x, :]
        else:
            start_x, stop_x = roi_x
            start_y, stop_y = roi_y
            rois = by_index[..., start_x:stop_x, start_y:stop_y]

        dark_run = dict.fromkeys(self.data_model.keys())

        if self._subtract_dark_cb.value:
            dark_run = {
                key: value.dark_data.image
                for key, value in self.data_model.items()}

        eval_ = partial(
            DataProcessing,
            path=path,
            pulse_ids=pulse_ids,
            train_index=train_index,
            rois=rois,
            )

        module_numbers = parse_ids(self._module_numbers.value)
        executor = ProcessPoolExecutor(max_workers=len(module_numbers))

        futures = {}
        for mod_no in module_numbers:
            futures[mod_no] = executor.submit(
                eval_, mod_no, dark_run=dark_run[mod_no])
            futures[mod_no].arg = mod_no
            futures[mod_no].add_done_callback(self.onProcessingDone)

        self._process_run.disabled=True

    @out.capture()
    def _on_fitting(self, e=None):

        if self.filtered is None:
            return

        params = []
        pid = self._memory_sl.value
        mod_no = self._module_dd.value

        bin_centers = self.data_model[mod_no].proc_data.st.bin_centers
        if pid > bin_centers.shape[0] - 1:
            return

        sigma = np.full((len(bin_centers[pid][self.peaks])), 10)
        params.extend(self.filtered[self.peaks])
        params.extend(sigma)
        params.extend(bin_centers[pid][self.peaks])

        try:
            fit_data, popt, perr = gauss_fit(
                bin_centers[pid], self.filtered, params)
        except Exception as ex:
            print(ex)
            return

        self._proc_hist_widget.data[1].x = bin_centers[pid]
        self._proc_hist_widget.data[1].y = fit_data

        table_elements = [
            f"{i:.2f}" + " +/- " + f"{j:.2f}" for i, j in zip(popt, perr)]
        table_elements = np.split(np.array(table_elements), 3)

        self._proc_fit_params_widget.data[0].cells.values = \
            [table_elements[2], table_elements[0], table_elements[1]]

    @out.capture()
    def onProcessingDone(self, future):
        if future.cancelled():
            print('{}: cancelled'.format(future.arg))
        elif future.done():
            error=future.exception()
            if error:
                print('{}: error returned: {}'.format(
                    future.arg, error))
            else:
                self.data_model[future.arg].proc_data.image = np.moveaxis(
                    future.result(), 0, 1)
                images = self.data_model[future.arg].proc_data.image
                centers_pr = []
                counts_pr = []
                def _eval_stat(pulse):
                    centers, counts = eval_statistics(
                        images[pulse, :, 0, ...], bins=1000)
                    return centers, counts

                with ThreadPoolExecutor(max_workers=10) as executor:
                    for ret in executor.map(_eval_stat, range(images.shape[0])):
                        centers_pr.append(ret[0])
                        counts_pr.append(ret[1])

                self.data_model[future.arg].proc_data.st.bin_centers = np.stack(
                    centers_pr)
                self.data_model[future.arg].proc_data.st.bin_counts = np.stack(
                    counts_pr)

                if future.arg == self._module_dd.value:
                    pid = self._memory_sl.value

                    self._proc_image_widget.data[0].z = np.mean(
                        images[pid, :, 0, ...], axis=0)
                    self._proc_hist_widget.data[0].x = \
                        self.data_model[future.arg].proc_data.st.bin_centers[pid]
                    self._proc_hist_widget.data[0].y = \
                        self.data_model[future.arg].proc_data.st.bin_counts[pid]

        self._process_run.disabled = False
        self._fitting_bt.disabled = False

    def clearPlots(self):
        pass

    def control_panel(self):
        display(self._visualization_widgets, self._cntrl, out)
