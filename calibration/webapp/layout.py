import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq

UPDATE_INT = 1.0

def get_dark_tab():
    return html.Div(
        className='control-tab',
        children=[
            html.Br(), html.Div([
                html.Div([
                    html.Label("Run Folder"),
                    dcc.Input(
                        id='run-folder',
                        placeholder="Enter dark run directory",
                        type='text',
                        value=""),

                    html.Hr(),
                    daq.BooleanSwitch(
                        id='load',
                        on=False),
                        ],
                        className="pretty_container one-third column"),
                    html.Div(id="load-info",
                             className="two-thirds column")], className="row")])


def get_general_setup_tab():
    div = html.Div(
        className='control-tab',
        children=[
        html.Br(),
        html.Div([
            html.Div([
                html.Label("Module Numbers", className="leftbox"),
                dcc.Input(
                    id='mod-no',
                    type='text',
                    value="14, 15",
                    className="rightbox"),

                html.Label("Pulse index", className="leftbox"),
                dcc.Input(
                    id='pulse-index',
                    type='text',
                    value="1:250:2",
                    className="rightbox"),

                html.Label("Roi X:", className="leftbox"),
                dcc.Input(
                    id='roi-x',
                    type='text',
                    value=":",
                    className="rightbox"),
                html.Label("Roi Y:", className="leftbox"),
                dcc.Input(
                    id='roi-y',
                    type='text',
                    value=":",
                    className="rightbox"),

                ],
                className="pretty_container one-third column"),
            html.Div(id="general-info",
                className="two-thirds column")], className="row")])

    return div


def get_plot_tab():
    return html.Div(id="visualization")


def get_layout():

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

        html.Br(),

        html.Div(children=[
            dcc.Tabs(
                parent_className='custom-tabs',
                className='custom-tabs-container',
                id='view-tabs',
                value='visualization',
                children=[
                    dcc.Tab(
                        className="custom-tab",
                        selected_className='custom-tab--selected',
                        label='General Set Up',
                        value='load-data',
                        children=get_general_setup_tab()
                    ),

                    dcc.Tab(
                        className="custom-tab",
                        selected_className='custom-tab--selected',
                        label='Dark Run',
                        value='dark-sub',
                        children=get_dark_tab()
                    ),

                    dcc.Tab(
                        className="custom-tab",
                        selected_className='custom-tab--selected',
                        label='Data Visualization',
                        value='visualization',
                        children=get_plot_tab()
                    )
                ])
        ])

    ])

    return app_layout
