import dash
from dash import html, dcc

dash.register_page(__name__, path='/')

layout = html.Div([
    html.H1('Home Page'),
    html.Div([
        dcc.Link(
            html.Button('Go to Eat', className='button'),
            href='/eat-dashboard'
        ),
        dcc.Link(
            html.Button('Go to Stats', className='button'),
            href='/stats-dashboard'
        )
    ], className='button-container')
])
