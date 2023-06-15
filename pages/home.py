import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/')
# TODO: finalize text blurbs or take them out
layout = html.Div(
    [
        dbc.Col(
            [
                dcc.Link(
                    html.Button('EAT Dashboard', className='button'),
                    href='/eat-dashboard',
                ),
                # html.P('Explore the EAT quantifer method, Add your own image sets, Test with TTI pre-generated datasets!', className='display-text'),
            ],
            className='button-link',
        ),
        dbc.Col(
            [
                dcc.Link(
                    html.Button('Label Dashboard', className='button'),
                    href='/label-dashboard',
                ),
                # html.P('Examine gender bias in profession, Label via more inclusive scales, Visualize your own image distributions', className='display-text'),
            ],
            className='button-link',
        ),
    ], className='button-container')
