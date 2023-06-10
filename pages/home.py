import dash
from dash import html, dcc

dash.register_page(__name__, path='/')

layout = html.Div(
    [
        dcc.Link(
            html.Button('EAT Dashboard', className='button'),
            href='/eat-dashboard',
            className='button-link',
        ),
        dcc.Link(
            html.Button('Stats Dashboard', className='button'),
            href='/stats-dashboard',
            className='button-link',
        )
    ], className='button-container')
