import os
import shutil
import atexit
import dash
from dash import html

app = dash.Dash(__name__, use_pages=True, external_stylesheets=['assets/styles.css'])
app.layout = dash.page_container


app.layout = html.Div([dash.page_container],
    style={'background-image': 'url("assets/background.png")', 'background-size': 'cover',
           'height': '100%', 'margin': 0, 'padding': 0},
)

def delete_local_files():
    if os.path.exists('upload'):
        shutil.rmtree('upload')

atexit.register(delete_local_files)

if __name__ == '__main__':
    # app.run_server(debug=True)
	app.run_server()
