import dash
from dash import html, dcc, callback, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

labor_stats = {'doctor': 43.8, 'nurse': 87.9, 'manager': 40.5, 'programmer': 22.1, 'financial analyst': 40.2, 'professor': 48.4}

options_menu = [
    {'label': 'doctor', 'value': 0},
    {'label': 'nurse', 'value': 1},
    {'label': 'manager', 'value': 2},
    {'label': 'programmer', 'value': 3},
    {'label': 'financial analyst', 'value': 4},
    {'label': 'professor', 'value': 5},
    {'label': 'rich', 'value': 6},
    {'label': 'poor', 'value': 7},
    {'label': 'assertive', 'value': 8},
    {'label': 'emotional', 'value': 9},
]


def histogram(figure, counts):
    x_fig = go.Histogram(
        x = sum([[i + 1] * count for i, count in enumerate(counts)], []),
        xbins=dict(start=1, end=11, size=1),
    )
    figure.add_trace(x_fig)
    return figure


dash.register_page(
    __name__,
    path='/stats-dashboard',
    title='Statistics Dashboard',
    name='Statistics Dashboard'
)
layout = html.Div([
    dcc.Link(
        html.Button('Go Home', className='button'),
        href='/',
        className="button-container",
    ),
    dbc.Row(
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'Stable Diffusion v1.4', 'value': 'sd'},
                {'label': 'Upload Your Own Images', 'value': 'upload'}
            ],
            value='-1',
            placeholder="Select the image source"
        ),
        className='dropdown-container',
    ),
    dbc.Row(
        [
            dbc.Col(
                dcc.Dropdown(
                    id='theme-dropdown',
                    options=options_menu,
                    value='-1',
                    placeholder="Select a theme",
                    className="dropdown-container",
                ),
                style={'width': '100%'},
            ),
            dbc.Col(
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Images', className="link-text")
                    ], className="display-text"),
                    className="upload-box",
                    multiple=True,
                ),
                style={'width': '100%'},
            ),
        ],
        className="row-container-even-dist",
    ),
    dbc.Row(
        [
            dbc.Col(
                dash_table.DataTable(
                    id="professions-table", 
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                ),
                style={'width': '25%', 'margin-left': '10px'},
            ),
            dbc.Col(
                dcc.Graph(id="histogram"),
                style={'width': '75%'},
            ),
        ],
        className="row-container-custom-dist",
    ),
])
@callback(
    Output('professions-table', 'data'),
    Output('histogram', 'figure'), 
    Input('model-dropdown', 'value'),
    Input('theme-dropdown', 'value'),
    Input('upload-image', 'contents'),
    Input('upload-image', 'filename'),
    # prevent_initial_call=True
)
def update_output(model_value, theme_value, contents, filenames):
    counts = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]    # TODO: counts of individual classes, get from model
    labor_stats_table = [{'occupation': key, 'female percentage': value} for key, value in labor_stats.items()]

    fig = go.Figure()
    fig.update_layout(
        title_text="Histogram", 
        xaxis=dict(
            title='Gender Expression Classes',
            tickvals=list(range(1, 11)),  # Specify the tick values
        ),
    )
    fig = histogram(fig, counts)
    # TODO: add scale image for reference once we finalize that
    return labor_stats_table, fig
