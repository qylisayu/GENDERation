import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/')
layout = html.Div(
    [
        dbc.Col(
            [
                dcc.Link(
                    html.Button('Embedding Association Test (EAT) Dashboard', className='button'),
                    href='/eat-dashboard',
                ),
            ],
            className='button-link',
        ),
        dbc.Col(
            [
                dcc.Link(
                    html.Button('MCAS & Gender Spectrum Labeling Dashboard', className='button'),
                    href='/label-dashboard',
                ),
            ],
            className='button-link',
        ),
        dbc.Col(
            [
                dcc.Link(
                    html.Button('GENDERation Report Website', className='button'),
                    href='https://genderation.github.io/',
                    target='_blank',
                ),
            ],
            className='button-link',
        ),
    ], className='button-container')
