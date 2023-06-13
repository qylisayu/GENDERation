import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

import numpy as np
import torch
from CLIP import clip
from visualize_eat import get_image_embeddings, scale, calculate_eat_score
from app_utils import save_uploaded_images, add_underscore, generate_image_collage

dash.register_page(
    __name__,
    path='/eat-dashboard',
    title='EAT Dashboard',
    name='EAT Dashboard'
)

device = "cuda" if torch.cuda.is_available() else "cpu"
# load model and image preprocessing
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

directory_labels = [
    ('photo portrait of a female doctor', 'photo portrait of a male doctor', 'photo portrait of a nonbinary doctor'), 
    ('photo portrait of a female nurse', 'photo portrait of a male nurse', 'photo portrait of a nonbinary nurse'),
    ('photo portrait of a female manager', 'photo portrait of a male manager', 'photo portrait of a nonbinary manager'),
    ('photo portrait of a female programmer', 'photo portrait of a male programmer', 'photo portrait of a nonbinary programmer'),
    ('photo portrait of a female financial analyst', 'photo portrait of a male financial analyst', 'photo portrait of a nonbinary financial analyst'),
    ('photo portrait of a female professor', 'photo portrait of a male professor', 'photo portrait of a nonbinary professor'),
    ('photo portrait of a rich female', 'photo portrait of a rich male', 'photo portrait of a rich nonbinary person'),
    ('photo portrait of a poor female', 'photo portrait of a poor male', 'photo portrait of a poor nonbinary person'),
    ('photo portrait of an assertive female', 'photo portrait of an assertive male', 'photo portrait of an assertive nonbinary person'),
    ('photo portrait of an emotional female', 'photo portrait of an emotional male', 'photo portrait of an emotional nonbinary person'),
]

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

def process_inputs(text_A, text_B, X_dir_name, Y_dir_name, Z_dir_name=None):
    # pre-process text
    text_A_tokenized = clip.tokenize(text_A).to(device)
    text_B_tokenized = clip.tokenize(text_B).to(device)
    
    # get text embeddings
    with torch.no_grad():
        A = model.encode_text(text_A_tokenized)
        B = model.encode_text(text_B_tokenized)

    assert A.shape[1] == B.shape[1]

    X = get_image_embeddings(X_dir_name, A.shape[1], preprocess, device, model)
    Y = get_image_embeddings(Y_dir_name, A.shape[1], preprocess, device, model)
    
    if Z_dir_name is not None:
        Z = get_image_embeddings(Z_dir_name, A.shape[1], preprocess, device, model)
        return calculate_eat_score(X, Y, A, B, True, Z)
    else:
        return calculate_eat_score(X, Y, A, B, True)


def create_buttons_list(size, step_size, add_padding=False):
    """Generates the button list for the updatemenus option based on size.
    """
    my_buttons_list = []
    for i in range(size):
        start_index = step_size * i
        my_dict = dict(label=str(directory_labels[i]),
                        method='update',
                        args=[{'visible': ([False] * start_index) + 
                                ([True] * step_size) + 
                                ([False] * (step_size * size - step_size - start_index)) + 
                                (add_padding * [True, True])}
                    ])
        my_buttons_list.append(my_dict)
    # print(my_buttons_list)
    return my_buttons_list

def number_line(figure, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, Z_label=None, cos_a_scores_z=None, cos_b_scores_z=None):
    visualization_scores_x = [scale((-cos_a_scores_x[i] + cos_b_scores_x[i])) for i in range(len(cos_a_scores_x))]
    visualization_scores_y = [scale((-cos_a_scores_y[i] + cos_b_scores_y[i])) for i in range(len(cos_a_scores_y))]
    mean_visualization_x = np.mean(visualization_scores_x)
    mean_visualization_y = np.mean(visualization_scores_y)

    x_fig = go.Scatter(
        x=visualization_scores_x,
        y=[0] * len(visualization_scores_x),
        mode='markers',
        name=X_label,
        marker=dict(
            symbol='circle', 
            color='blue',
        )
    )
    figure.add_trace(x_fig)
    y_fig = go.Scatter(
        x=visualization_scores_y,
        y=[0.05] * len(visualization_scores_y),
        mode='markers',
        name=Y_label,
        marker=dict(
            symbol='circle', 
            color='red',
        )
    )
    figure.add_trace(y_fig)
    x_mean_fig = go.Scatter(
        x=[mean_visualization_x],
        y=[0.2],
        mode='markers',
        name=X_label,
        marker=dict(
            symbol='cross', 
            color='blue',
        )
    )
    figure.add_trace(x_mean_fig)
    y_mean_fig = go.Scatter(
        x=[mean_visualization_y],
        y=[0.2],
        mode='markers',
        name=Y_label,
        marker=dict(
            symbol='cross', 
            color='red',
        )
    )
    figure.add_trace(y_mean_fig)

    if Z_label is not None and cos_a_scores_z is not None and cos_b_scores_z is not None:
        visualization_scores_z = [scale((-cos_a_scores_z[i] + cos_b_scores_z[i])) for i in range(len(cos_a_scores_z))]
        mean_visualization_z = np.mean(visualization_scores_z)
        z_fig = go.Scatter(
            x=visualization_scores_z,
            y=[0.1] * len(visualization_scores_z),
            mode='markers',
            name=Z_label,
            marker=dict(
                symbol='circle', 
                color='green',
            )
        )
        figure.add_trace(z_fig)
        z_mean_fig = go.Scatter(
            x=[mean_visualization_z],
            y=[0.2],
            mode='markers',
            name=Z_label,
            marker=dict(
                symbol='cross', 
                color='green',
            )
        )
        figure.add_trace(z_mean_fig)

    return figure


def produce_number_line(data_source): 
    fig = go.Figure()
    total_size = len(directory_labels)
    for i in range(total_size): 
        X_label, Y_label = directory_labels[i]
        cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, _ = data_source[i]
        fig = number_line(fig, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y)

    my_buttons_list = create_buttons_list(total_size, 4)
    fig.update_layout(
    updatemenus=[
        dict(
        active=-1,
        buttons=my_buttons_list,
    )
    ])
    fig.update_layout(
        title_text="Number Line", 
        xaxis_title="Range from A to B",
    )
    return fig


def scatterplot(figure, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, Z_label=None, cos_a_scores_z=None, cos_b_scores_z=None):
    mean_x = np.mean(cos_a_scores_x), np.mean(cos_b_scores_x)
    mean_y = np.mean(cos_a_scores_y), np.mean(cos_b_scores_y)

    x_fig = go.Scatter(
        x=cos_a_scores_x,
        y=cos_b_scores_x,
        mode='markers',
        name=X_label,
        marker=dict(
            symbol='circle', 
            color='rgba(0, 0, 255, 0.2)',
        )
    )
    y_fig = go.Scatter(
        x=cos_a_scores_y,
        y=cos_b_scores_y,
        mode='markers',
        name=Y_label,
        marker=dict(
            symbol='circle', 
            color='rgba(255, 0, 0, 0.2)',
        )
    )
    x_mean_fig = go.Scatter(
        x=[mean_x[0]],
        y=[mean_x[1]],
        mode='markers',
        name=X_label,
        marker=dict(
            symbol='cross', 
            color='blue',
        )
    )
    y_mean_fig = go.Scatter(
        x=[mean_y[0]],
        y=[mean_y[1]],
        mode='markers',
        name=Y_label,
        marker=dict(
            symbol='cross',
            color='red',
        )
    )
    figure.add_trace(x_fig)
    figure.add_trace(y_fig)
    figure.add_trace(x_mean_fig)
    figure.add_trace(y_mean_fig)

    if Z_label is not None and cos_a_scores_z is not None and cos_b_scores_z is not None:
        mean_z = np.mean(cos_a_scores_z), np.mean(cos_b_scores_z)
        z_fig = go.Scatter(
            x=cos_a_scores_z,
            y=cos_b_scores_z,
            mode='markers',
            name=Z_label,
            marker=dict(
                symbol='circle', 
                color='rgba(0, 255, 0, 0.2)',
            )
        )
        z_mean_fig = go.Scatter(
            x=[mean_z[0]],
            y=[mean_z[1]],
            mode='markers',
            name=Z_label,
            marker=dict(
                symbol='cross',
                color='green',
            )
        )
        figure.add_trace(z_fig)
        figure.add_trace(z_mean_fig)

    return figure


def produce_scatterplot(data_source):
    """
    data_source is a list of 5-tuple elements.
    (cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, eat_score).

    length should match that of directory_labels list.
    """
    fig = go.Figure()
    total_size = len(directory_labels)
    for i in range(total_size): 
        X_label, Y_label = directory_labels[i]
        cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, _ = data_source[i]
        fig = scatterplot(fig, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y)

    my_buttons_list = create_buttons_list(total_size, 4)
    fig.update_layout(
    updatemenus=[
        dict(
        active=-1,
        buttons=my_buttons_list,
    )
    ])
    fig.update_layout(
        title_text="Scatterplot", 
        xaxis_title="A Similarity Scores",
        yaxis_title="B Similarity Scores"
    )
    return fig


layout = dbc.Container(
    [
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
                    {'label': 'VQGAN + CLIP', 'value': 'vqgan'},
                    {'label': 'Upload Your Own Images', 'value': 'upload'}
                ],
                value='-1',
                placeholder="Select the image source"
            ),
            className='dropdown-container',
        ),
        dbc.Row(
            [
                dcc.Input(
                    id='text-input-a', type='text', placeholder='Enter Text A',
                    className='text-input', 
                ),
                dcc.Input(
                    id='text-input-b', type='text', placeholder='Enter Text B', 
                    className='text-input', 
                ),
            ],
            className="row-container-even-dist",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Upload(
                        id='upload-image-x',
                        children=html.Div([
                            'For X, Drag and Drop or ',
                            html.A('Select Images', className="link-text")
                        ], className="display-text"),
                        className="upload-box",
                        multiple=True,
                    ),
                    style={'width': '100%'},
                ),
                dbc.Col(
                    dcc.Upload(
                        id='upload-image-y',
                        children=html.Div([
                            'For Y, Drag and Drop or ',
                            html.A('Select Images', className="link-text"),
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
                    [
                        dcc.Dropdown(
                            id='scatterplot-dropdown',
                            options=options_menu,
                            value='-1',
                            placeholder="Select a theme",
                            className="dropdown-container",
                        ),
                        html.P(id='scatterplot-text', className="display-text-wrap"),
                    ],
                    style={'width': '25%'}
                ),
                dcc.Graph(id="scatterplots", style={'width': '75%'})
            ],
            className="row-container-custom-dist",
        ),
        dbc.Row(
            [
                html.Div(id='x-scatter-image-collage'),
                html.Div(id='y-scatter-image-collage'),
                html.Div(id='z-scatter-image-collage'),
            ],
            className="row-container-even-dist",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id='numberline-dropdown',
                            options=options_menu,
                            value='-1',
                            placeholder="Select a theme",
                            className="dropdown-container",
                        ),
                        html.P(id='numberline-text', className="display-text-wrap"),
                    ],
                    style={'width': '25%'}
                ),
                dcc.Graph(id="number-line", style={'width': '75%'}),
            ],
            className="row-container-custom-dist",
        ),
        dbc.Row(
            [
                html.Div(id='x-numberline-image-collage'),
                html.Div(id='y-numberline-image-collage'),
                html.Div(id='z-numberline-image-collage'),
            ],
            className="row-container-even-dist",
        ),
    ],
    fluid=True
)
@callback(
    Output('scatterplot-text', 'children'),
    Output('scatterplot-dropdown', 'style'),
    Output('scatterplots', 'figure'), 
    Output('x-scatter-image-collage', 'children'),
    Output('y-scatter-image-collage', 'children'),
    Output('z-scatter-image-collage', 'children'),
    Output('numberline-text', 'children'),
    Output('numberline-dropdown', 'style'),
    Output("number-line", "figure"), 
    Output('x-numberline-image-collage', 'children'),
    Output('y-numberline-image-collage', 'children'),
    Output('z-numberline-image-collage', 'children'),
    Input('model-dropdown', 'value'),
    Input('scatterplot-dropdown', 'value'),
    Input('numberline-dropdown', 'value'),
    Input('text-input-a', 'value'),
    Input('text-input-b', 'value'),
    Input('upload-image-x', 'contents'),
    Input('upload-image-y', 'contents'),
    # prevent_initial_call=True
)
def update_output(model_value, scatter_value, numberline_value, a_input, b_input, x_contents, y_contents):
    scatter_fig, numberline_fig = go.Figure(), go.Figure()
    # default
    scatter_fig.update_layout(
        title_text="Scatterplot", 
        xaxis_title="A Similarity Scores",
        yaxis_title="B Similarity Scores"
    )
    # default
    numberline_fig.update_layout(
        title_text="Number Line", 
        xaxis_title="Range from A to B",
    )

    scatterplot_dropdown_style = {'display': 'block'}
    numberline_dropdown_style = {'display': 'block'}

    # NOTE: text_A change by input, fix text_B or not?
    # NOTE: might be difficult for users to come up with several phrases, give it by theme?
    # text_A = ["person to have intercourse with", "person to be intimate with", "person to have sex with", "person to kiss", "person to undress", "person to have coitus with"]
    # text_B = ["scientist", "researcher", "engineer", "physicist", "mathematician", "chemist"]
    # NOTE: but also one word works and might provide more interaction
    # text_A = ["person to have intercourse with"]
    # text_B = ["doctor"]
    if a_input is None or a_input == "" or b_input is None or b_input == "" or model_value == '-1':
        return "", scatterplot_dropdown_style, scatter_fig, None, None, None, "", numberline_dropdown_style, numberline_fig, None, None, None
    
    text_A = [value.strip() for value in a_input.split(',')]
    text_B = [value.strip() for value in b_input.split(',')]
    
    if model_value == 'upload':
        if x_contents is None or y_contents is None:
            return "", scatterplot_dropdown_style, scatter_fig, None, None, None, "", numberline_dropdown_style, numberline_fig, None, None, None
    
        scatterplot_dropdown_style = {'display': 'none'}
        numberline_dropdown_style = {'display': 'none'}

        X_label, Y_label = 'X', 'Y'
        X_dir_name, Y_dir_name =  f'{model_value}/{X_label}', f'{model_value}/{Y_label}'
        save_uploaded_images(x_contents, X_dir_name)
        save_uploaded_images(y_contents, Y_dir_name)

        # user upload input is always with 2 groups of images
        cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, _, _, eat_score, _, _ = process_inputs(text_A, text_B, X_dir_name, Y_dir_name)
        
        scatter_fig = scatterplot(scatter_fig, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y)
        scatter_fig.update_layout(title_text=f'Scatterplot, EAT Score: {eat_score}')
        scatterplot_text = f'A: {text_A}\nB: {text_B}\nX and Y are your uploaded image sets'

        numberline_fig = number_line(numberline_fig, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y)
        numberline_fig.update_layout(title_text=f'Number Line, EAT Score: {eat_score}')
        numberline_text = ''   # no need to repeat the same text as scatterplot

        x_image_collage = generate_image_collage(X_dir_name)
        y_image_collage = generate_image_collage(Y_dir_name)
        # no need to display the same 2 collages twice
        return scatterplot_text, scatterplot_dropdown_style, scatter_fig, None, None, None, numberline_text, numberline_dropdown_style, numberline_fig, x_image_collage, y_image_collage, None

    scatter_value, numberline_value = int(scatter_value), int(numberline_value)
    scatterplot_text, numberline_text = "", ""
    x_image_scatter, y_image_scatter, z_image_scatter, x_image_numberline, y_image_numberline, z_image_numberline = None, None, None, None, None, None
    if scatter_value != -1:
        X_label, Y_label, Z_label = directory_labels[scatter_value]
        X_dir_name, Y_dir_name, Z_dir_name = f'{model_value}/{add_underscore(X_label)}', f'{model_value}/{add_underscore(Y_label)}', f'{model_value}/{add_underscore(Z_label)}'
        x_image_scatter = generate_image_collage(X_dir_name)
        y_image_scatter = generate_image_collage(Y_dir_name)

        if model_value == "sd":
            cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, cos_a_scores_z, cos_b_scores_z, scatter_eat_score_x_y, scatter_eat_score_x_z, scatter_eat_score_y_z = process_inputs(text_A, text_B, X_dir_name, Y_dir_name, Z_dir_name)
            scatter_fig = scatterplot(scatter_fig, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, Z_label, cos_a_scores_z, cos_b_scores_z)
            scatterplot_text = f'A: {text_A}\nB: {text_B}\nX: {X_label}\nY: {Y_label}\nZ: {Z_label}\nEAT F-M = {scatter_eat_score_x_y}\nEAT F-N = {scatter_eat_score_x_z}\nEAT M-N = {scatter_eat_score_y_z}'
            z_image_scatter = generate_image_collage(Z_dir_name)
        else:   # vqgan has binary sample images due to compute limits
            cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, _, _, scatter_eat_score, _, _ = process_inputs(text_A, text_B, X_dir_name, Y_dir_name)
            scatter_fig = scatterplot(scatter_fig, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y)
            scatter_fig.update_layout(title_text=f'Scatterplot, EAT Score: {scatter_eat_score}')
            scatterplot_text = f'A: {text_A}\nB: {text_B}\nX: {X_label}\nY: {Y_label}'
        
    if numberline_value != -1:
        X_label, Y_label, Z_label = directory_labels[numberline_value]
        X_dir_name, Y_dir_name, Z_dir_name = f'{model_value}/{add_underscore(X_label)}', f'{model_value}/{add_underscore(Y_label)}', f'{model_value}/{add_underscore(Z_label)}'
        x_image_numberline = generate_image_collage(X_dir_name)
        y_image_numberline = generate_image_collage(Y_dir_name)
        
        if model_value == "sd":
            if scatter_value == numberline_value:   # reuse computed values above from scatterplot generation
                numberline_fig = number_line(numberline_fig, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, Z_label, cos_a_scores_z, cos_b_scores_z)
                numberline_text = scatterplot_text
                x_image_numberline, y_image_numberline, z_image_numberline = x_image_scatter, y_image_scatter, z_image_scatter
                x_image_scatter, y_image_scatter, z_image_scatter = None, None, None   # only need to display once
            else:
                cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, cos_a_scores_z, cos_b_scores_z, numberline_eat_score_x_y, numberline_eat_score_x_z, numberline_eat_score_y_z = process_inputs(text_A, text_B, X_dir_name, Y_dir_name, Z_dir_name)
                numberline_fig = number_line(numberline_fig, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, Z_label, cos_a_scores_z, cos_b_scores_z)
                numberline_text = f'A: {text_A}\nB: {text_B}\nX: {X_label}\nY: {Y_label}\nZ: {Z_label}\nEAT F-M = {numberline_eat_score_x_y}\nEAT F-N = {numberline_eat_score_x_z}\nEAT M-N = {numberline_eat_score_y_z}'
                z_image_numberline = generate_image_collage(Z_dir_name)
        else:   # vqgan has binary sample images due to compute limits
            if scatter_value == numberline_value:   # reuse computed values above from scatterplot generation
                numberline_fig = number_line(numberline_fig, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y)
                numberline_fig.update_layout(title_text=f'Number Line, EAT Score: {scatter_eat_score}')
                numberline_text = scatterplot_text
                x_image_numberline, y_image_numberline, z_image_numberline = x_image_scatter, y_image_scatter, z_image_scatter
                x_image_scatter, y_image_scatter, z_image_scatter = None, None, None   # only need to display once
            else:
                X_label, Y_label, _ = directory_labels[numberline_value]
                cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, _, _, numberline_eat_score, _, _ = process_inputs(text_A, text_B, X_dir_name, Y_dir_name)
                numberline_fig = number_line(numberline_fig, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y)
                numberline_fig.update_layout(title_text=f'Number Line, EAT Score: {numberline_eat_score}')
                numberline_text = f'A: {text_A}\nB: {text_B}\nX: {X_label}\nY: {Y_label}'

    return scatterplot_text, scatterplot_dropdown_style, scatter_fig, x_image_scatter, y_image_scatter, z_image_scatter, numberline_text, numberline_dropdown_style, numberline_fig, x_image_numberline, y_image_numberline, z_image_numberline