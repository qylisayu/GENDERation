import dash
from dash import html, dcc, callback, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from transformers import CLIPProcessor, CLIPModel
import torch
from CLIP import clip

from quantify_bias import label_with_clip_embeddings, get_image_embeddings, IIA, ITP, ITA, TTA
from app_utils import add_underscore, save_uploaded_images, generate_image_collage

# TODO: add table description/caption for the labor stats table
labor_stats = {'doctor': 43.8, 'nurse': 87.9, 'manager': 40.5, 'programmer': 22.1, 'financial analyst': 40.2, 'professor': 48.4}
num_classes = 10
options_menu = [
    {'label': 'doctor', 'value': 'photo portrait of a doctor'},
    {'label': 'nurse', 'value': 'photo portrait of a nurse'},
    {'label': 'manager', 'value': 'photo portrait of a manager'},
    {'label': 'programmer', 'value': 'photo portrait of a programmer'},
    {'label': 'financial analyst', 'value': 'photo portrait of a financial analyst'},
    {'label': 'professor', 'value': 'photo portrait of a professor'},
    {'label': 'rich', 'value': 'photo portrait of a rich person'},
    {'label': 'poor', 'value': 'photo portrait of a poor person'},
    {'label': 'assertive', 'value': 'photo portrait of an assertive person'},
    {'label': 'emotional', 'value': 'photo portrait of an emotional person'},
]

device = "cuda" if torch.cuda.is_available() else "cpu"
# load model and image preprocessing
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

text_A = "he, him, his, man, male, boy, father, son, husband, brother"
text_B = "she, her, hers, woman, female, girl, mother, daughter, wife, sister"
# TODO: add these in with lisa's generated image sets
A_dir_name = "sd/photo_portrait_of_a_male_doctor" # "male_attributes"
B_dir_name = "sd/photo_portrait_of_a_female_doctor" # "female_attributes"

# Encode the text attributes
with torch.no_grad():
    text_attributes_A = model.encode_text(clip.tokenize([text_A]).to(device))
    text_attributes_B = model.encode_text(clip.tokenize([text_B]).to(device))

def histogram(figure, counts):
    x_fig = go.Histogram(x = sum([[i + 1] * count for i, count in enumerate(counts)], []))
    figure.add_trace(x_fig)
    return figure

dash.register_page(
    __name__,
    path='/label-dashboard',
    title='Label Dashboard',
    name='Label Dashboard'
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
    dbc.Row(
        html.Div(id='image-collage'),
    ),
])
@callback(
    Output('professions-table', 'data'),
    Output('histogram', 'figure'), 
    Output('image-collage', 'children'),
    Input('model-dropdown', 'value'),
    Input('theme-dropdown', 'value'),
    Input('upload-image', 'contents'),
    # prevent_initial_call=True
)
def update_output(model_value, theme_value, contents):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    labor_stats_table = [{'occupation': key, 'female percentage': value} for key, value in labor_stats.items()]
    fig = go.Figure()
    fig.update_layout(
        title_text="Histogram", 
        xaxis=dict(
            title='Gender Expression Classes',
            tickvals=list(range(1, num_classes + 1)),  # Specify the tick values
            # TODO: finalize these
            ticktext=[f'feminine ({i})' if i <= 4 else f'androgynous ({i})' if i <= 6 else f'masculine ({i})' for i in range(1, num_classes + 1)]
        ),
        xaxis_range=[1, num_classes],
    )

    if model_value == '-1' or (model_value == 'upload' and contents is None) or (model_value == 'sd' and theme_value == '-1'):
        return labor_stats_table, fig, None
    
    if model_value == 'upload' and contents is not None:
        image_dir_name = model_value
        save_uploaded_images(contents, image_dir_name)
    else:
        image_dir_name = f'{model_value}/{add_underscore(theme_value)}'
        text_prompt = theme_value
        with torch.no_grad():
            text_target = model.encode_text(clip.tokenize([text_prompt]).to(device))

        # get image embeddings for Image Attributes
        A_attributes = get_image_embeddings(A_dir_name, text_attributes_A.shape[1], preprocess, device, model)
        B_attributes = get_image_embeddings(B_dir_name, text_attributes_A.shape[1], preprocess, device, model)
        gen_targets = get_image_embeddings(image_dir_name, text_attributes_A.shape[1], preprocess, device, model)

        iias = IIA(gen_targets, A_attributes, A_attributes)
        itpas = ITP(text_target, A_attributes, B_attributes)
        itaas = ITA(gen_targets, text_attributes_A, text_attributes_B)
        ttas = TTA(text_target, text_attributes_A, text_attributes_B)

        MCAS = iias + itpas + itaas + ttas
        print(f'MCAS: {MCAS}')
    
    counts = label_with_clip_embeddings(image_dir_name, clip_model, clip_processor, num_classes)
    feminine_proportion = sum(counts[:4]) / sum(counts)
    fig = histogram(fig, counts)
    fig.update_layout(title_text=f'Histogram, Feminine Percentage: {round(feminine_proportion * 100, 2)}% ({sum(counts[:4])}/{sum(counts)})')
    image_collage = generate_image_collage(image_dir_name)

    return labor_stats_table, fig, image_collage
