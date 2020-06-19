"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import plotly.graph_objects as go


class ScatterPlot(go.Figure):
    """ScatterPlot class"""
    def __init__(self,
                 title=None,
                 xlabel=None,
                 ylabel=None,
                 legend='',
                 drop_down_label='',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Attributes
        ----------
        _title: (str) Title of the plot
        _xlabel: (str) xlabel for the scatter plot
        _ylabel: (str) ylabel for the scatter plot
        _legend: (str) Prefix to legend that will be used to label each plot
        _drop_down_label: (str) Prefix to Drow down option that will be used to
            label 2 dimension of ydata in case ydata is 3 dimensional
        """
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._legend = legend
        self._drop_down_label = drop_down_label
        self.update_layout(
            title=self._title,
            title_x=0.5,
            xaxis=dict(title=self._xlabel),
            yaxis=dict(title=self._ylabel))

    def setData(self, xdata, ydata, yerror=None):
        """
        xdata: 1d numpy array
        ydata: np.ndarray (1, 2 or 3 dimensional)
            If 3 dimensional: Second dimension will be used for Drop down
        yerror: np.ndarray. Same shape as ydata
            Optional

        Raises:
            if ydata is more than 3 dimensional or first dimension does
            not match xdata
        """
        if len(ydata.shape) not in [1, 2, 3] or \
            xdata.shape[0] != ydata.shape[0]:
            raise

        if yerror is not None and yerror.shape != ydata.shape:
            raise

        traces = []
        if len(ydata.shape) == 1:
            data = go.Scatter(
                x=xdata,
                y=ydata,
                mode='lines+markers')

            if yerror is not None:
                data.error_y=dict(
                    type='data',
                    array=yerror,
                    visible=True)
            traces.append(data)

        elif len(ydata.shape) == 2:

            for idx in range(ydata.shape[1]):
                data = go.Scatter(
                    x=xdata,
                    y=ydata[:, idx],
                    mode='lines+markers',
                    name=f"{self._legend} {idx}")

                if yerror is not None:
                    data.error_y=dict(
                        type='data',
                        array=yerror[:, idx],
                        visible=True)
                traces.append(data)

        else:
            shape = ydata.shape
            for n, index_1 in enumerate(range(shape[-1])):
                for index_2 in range(shape[-2]):
                    data = go.Scatter(
                        x=xdata,
                        y=ydata[:, index_2, index_1],
                        visible = False,
                        mode='lines+markers',
                        name=f"{self._legend}: {index_1}")

                    if yerror is not None:
                        data.error_y=dict(
                            type='data',
                            array=yerror[:, index_2, index_1],
                            visible=True)
                    traces.append(data)
                traces[n * shape[-2]].visible = True

            options_dd = []
            for index_2 in range(shape[-2]):
                visible = [False] * shape[-2]
                visible[index_2] = True
                temp_dict = dict(
                    label = str(f"{self._drop_down_label}: {index_2}"),
                    method = 'update',
                    args = [{'visible': visible}])
                options_dd.append(temp_dict)

            updatemenus = [dict(
                buttons=options_dd,
                direction="down",
                showactive=True,
                x=0.,
                xanchor="right",
                y=1,
                yanchor="top")]
            self.update_layout(updatemenus=updatemenus)

        for trace in traces:
            self.add_trace(trace)
