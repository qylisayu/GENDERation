import dash
from dash import html, dcc, callback, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from transformers import CLIPProcessor, CLIPModel
import os
import shutil
import base64
from visualize_eat import label_with_clip_embeddings

labor_stats = {'doctor': 43.8, 'nurse': 87.9, 'manager': 40.5, 'programmer': 22.1, 'financial analyst': 40.2, 'professor': 48.4}
num_classes = 10

options_menu = [
    {'label': 'doctor', 'value': 'photo_portrait_of_a_doctor'},
    {'label': 'nurse', 'value': 'photo_portrait_of_a_nurse'},
    {'label': 'manager', 'value': 'photo_portrait_of_a_manager'},
    {'label': 'programmer', 'value': 'photo_portrait_of_a_programmer'},
    {'label': 'financial analyst', 'value': 'photo_portrait_of_a_financial_analyst'},
    {'label': 'professor', 'value': 'photo_portrait_of_a_professor'},
    {'label': 'rich', 'value': 'photo_portrait_of_a_rich_person'},
    {'label': 'poor', 'value': 'photo_portrait_of_a_poor_person'},
    {'label': 'assertive', 'value': 'photo_portrait_of_an_assertive_person'},
    {'label': 'emotional', 'value': 'photo_portrait_of_an_emotional_person'},
]


def histogram(figure, counts):
    x_fig = go.Histogram(
        x = sum([[i + 1] * count for i, count in enumerate(counts)], []),
        xbins=dict(start=1, end=11, size=1),
    )
    figure.add_trace(x_fig)
    return figure


def save_uploaded_images(contents, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:   # account for reupload case, where old number of images > new number of images
        shutil.rmtree(directory)
        os.makedirs(directory)

    for i in range(len(contents)):
        content = contents[i]

        # Get the image data
        _, content_string = content.split(',')

        # Decode and save the image locally
        decoded_image = base64.b64decode(content_string)
        image_filename = os.path.join(f'{directory}', f'{i}.png')
        with open(image_filename, 'wb') as f:
            f.write(decoded_image)


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
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    labor_stats_table = [{'occupation': key, 'female percentage': value} for key, value in labor_stats.items()]
    fig = go.Figure()
    fig.update_layout(
        title_text="Histogram", 
        xaxis=dict(
            title='Gender Expression Classes',
            tickvals=list(range(1, num_classes + 1)),  # Specify the tick values
            ticktext=['feminine' if i <= 4 else 'androgynous' if i <= 6 else 'masculine' for i in range(1, num_classes + 1)]  # TODO: finalize these
        ),
        xaxis_range=[1, num_classes],
    )

    if model_value == '-1' or (model_value == 'upload' and contents is None) or (model_value == 'sd' and theme_value == '-1'):
        return labor_stats_table, fig
    
    if model_value == 'upload' and contents is not None:
        image_dir_name = model_value
        save_uploaded_images(contents, image_dir_name)
    else:
        image_dir_name = f'{model_value}/{theme_value}'
    counts = label_with_clip_embeddings(image_dir_name, model, processor, num_classes)
    fig = histogram(fig, counts)

    return labor_stats_table, fig
