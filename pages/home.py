import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(
    __name__, 
    path='/',
    title='GENDERation Home',
    name='GENDERation Home'
)
layout = dbc.Container([
    html.Img(
        src='assets/logo_white.png', alt='GENDeation Logo', 
        style={'margin-left': '20%', 'margin-top': '8%'}
    ),
    dbc.Row(
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
        ], 
        className='button-container',
    )
])