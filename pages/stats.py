import dash
from dash import dcc, html

dash.register_page(
    __name__,
    path='/stats-dashboard',
    title='Statistics Dashboard',
    name='Statistics Dashboard'
)
layout = html.Div([
    dcc.Link(
        html.Button('Go Home', className='button'),
        href='/'
    ),
])