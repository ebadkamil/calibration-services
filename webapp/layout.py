import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq
from utils import get_virtual_memory

_VIRTUAL, _SWAP = get_virtual_memory()

colors_map = ['Blackbody', 'Reds', 'Viridis']

_SOURCE = {
    "LPD": ["FXE_DET_LPD1M-1/DET/detector"],
    "AGIPD": ["SPB_DET_AGIPD1M-1/DET/detector"]
}


def get_dark_tab(config):
    return html.Div(className='control-tab',
                    children=[
                        html.Br(), html.Div([

                            html.Div([
                                html.Label("Run Folder"),
                                dcc.Input(
                                    id='run-folder',
                                    placeholder="Enter run directory",
                                    type='text',
                                    value=config["run_folder"]),

                                html.Hr(),
                                daq.BooleanSwitch(
                                    id='load',
                                    on=False
                                ),
                            ],
                                className="pretty_container one-third column"),
                            html.Div(id="load-info",
                                     className="two-thirds column")], className="row")])


def get_exp_tab():
    return html.Div(id="experimental-params")


def get_plot_tab(config):
    div = html.Div(
        children=[html.Br(),
            html.Div([
                html.Div(
                    [html.Label("Calibration Analysis Setup"),
                     html.Hr(),
                     html.Label("Pixel size:", className="leftbox"),
                     dcc.Input(
                        id='pixel-size',
                        type='number',
                        value=config["pixel_size"],
                        className="rightbox"),
                     html.Label("Mask range:", className="leftbox"),
                     dcc.RangeSlider(
                        id='mask-rng',
                        min=0,
                        max=10000.0,
                        value=config["mask_rng"],
                        className="rightbox"),
                     html.Label("Geometry:", className="leftbox"),
                     dcc.Input(
                        id='geom-file',
                        type='text',
                        value=config["geom_file"],
                        className="rightbox"),

                     ], className="pretty_container six columns"),
                html.Div([
                    html.Label("Source:", className="leftbox"),
                    dcc.Dropdown(
                        id='source',
                        options=[{'label': i, 'value': i} for i in config["source_name"]],
                        value=config["source_name"][0]),
                    html.Hr(),
                    html.Label("Fitting Function:", className="leftbox"),
                    dcc.Dropdown(
                        id='fitting-type',
                        options=[{'label': i, 'value': i}
                                 for i in ["Gaussian", "Lorentizian"]],
                        value="Gaussian",
                        className="rightbox"),
                    html.Label("Projection:", className="leftbox"),
                    dcc.Dropdown(
                        id='roi-projection',
                        options=[{'label': i, 'value': f"projection_{i}"} for i in ['x', 'y']],
                        value="projection_x",
                        className="rightbox"),
                    html.Div(id="logger")
                ], className="pretty_container six columns")

            ], className="row"),
            html.Div(
            [html.Div(
                [dcc.Dropdown(
                    id='color-scale',
                    options=[{'label': i, 'value': i}
                             for i in colors_map],
                    value=colors_map[0])],
                className="pretty_container six columns"),
             html.Div(
                [html.Label("Pulses: ", className="leftbox"),
                 dcc.Slider(
                    id='n-pulses',
                    min=1,
                    max=400,
                    value=10,
                    step=1,
                    className="rightbox")],
                className="pretty_container six columns")],
            className="row"),

            html.Div([
                html.Div(
                    [dcc.Graph(
                        id='mean-image')],
                    className="pretty_container six columns"),
                html.Div(
                    [dcc.Graph(
                        id='image-histogram')],
                    className="pretty_container six columns")],
            className="row"),
        ])
    return div

def get_layout(UPDATE_INT, config=None):

    app_layout = html.Div([

        html.Div([
            dcc.Interval(
                id='psutil_component',
                interval=UPDATE_INT * 1000,
                n_intervals=0)], style=dict(textAlign='center')),

        html.Div([
            daq.Gauge(
                id='virtual_memory',
                min=0,
                value=0,
                size=150,
                className="leftbox",
                style=dict(textAlign="center")),
            daq.Gauge(
                id='swap_memory',
                min=0,
                value=0,
                size=150,
                className="rightbox",
                style=dict(textAlign="center")
            ),
        ]),
        daq.LEDDisplay(
            id='train-id',
            value=1000,
            color="#FF5E5E",
            style=dict(textAlign="center")),

        html.Br(),

        html.Div(children=[
            dcc.Tabs(
                parent_className='custom-tabs',
                className='custom-tabs-container',
                id='view-tabs',
                value='plot',
                children=[
                    dcc.Tab(
                        className="custom-tab",
                        selected_className='custom-tab--selected',
                        label='Load data',
                        value='load-data',
                        children=get_dark_tab(config)
                    ),

                    # dcc.Tab(
                    #     className="custom-tab",
                    #     selected_className='custom-tab--selected',
                    #     label='Experimental parameters',
                    #     value='exp-param',
                    #     children=get_exp_tab()
                    # ),

                    dcc.Tab(
                        className="custom-tab",
                        selected_className='custom-tab--selected',
                        label='Plots',
                        value='plot',
                        children=get_plot_tab(config)
                    )
                ])
        ])

    ])

    return app_layout
