import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import numpy as np
import torch
from CLIP import clip
from visualize_eat import get_image_embeddings, scale, calculate_eat_score

device = "cuda" if torch.cuda.is_available() else "cpu"
# load model and image preprocessing
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

X_Y_pairs = [
    ('photo portrait of a female doctor', 'photo portrait of a male doctor'), 
    ('photo portrait of a female nurse', 'photo portrait of a male nurse'),
    ('photo portrait of a female manager', 'photo portrait of a male manager'),
    ('photo portrait of a female programmer', 'photo portrait of a male programmer'),
    ('photo portrait of a female financial analyst', 'photo portrait of a male financial analyst'),
    ('photo portrait of a female professor', 'photo portrait of a male professor'),
    ('photo portrait of a rich female', 'photo portrait of a rich male'),
    ('photo portrait of a poor female', 'photo portrait of a poor male'),
    ('photo portrait of an assertive female', 'photo portrait of an assertive male'),
    ('photo portrait of an emotional female', 'photo portrait of an emotional male'),
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

def add_underscore(string_to_modify):
    return str.replace(string_to_modify, ' ', '_')


def process_inputs(text_A, text_B, X_dir_name, Y_dir_name):
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
    return calculate_eat_score(X, Y, A, B)


def create_buttons_list(size, step_size, add_padding=False):
    """Generates the button list for the updatemenus option based on size.
    """
    my_buttons_list = []
    for i in range(size):
        start_index = step_size * i
        my_dict = dict(label=str(X_Y_pairs[i]),
                        method='update',
                        args=[{'visible': ([False] * start_index) + 
                                ([True] * step_size) + 
                                ([False] * (step_size * size - step_size - start_index)) + 
                                (add_padding * [True, True])}
                    ])
        my_buttons_list.append(my_dict)
    # print(my_buttons_list)
    return my_buttons_list


def number_line(figure, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y):
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
        y=[0] * len(visualization_scores_y),
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
        y=[1],
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
        y=[1],
        mode='markers',
        name=Y_label,
        marker=dict(
            symbol='cross', 
            color='red',
        )
    )
    figure.add_trace(y_mean_fig)

    return figure


def produce_number_line(data_source): 
    fig = go.Figure()
    total_size = len(X_Y_pairs)
    for i in range(total_size): 
        X_label, Y_label = X_Y_pairs[i]
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


def scatterplot(figure, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y):
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

    return figure


def produce_scatterplot(data_source):
    """
    data_source is a list of 5-tuple elements.
    (cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, eat_score).

    length should match that of X_Y_pairs list.
    """
    fig = go.Figure()
    total_size = len(X_Y_pairs)
    for i in range(total_size): 
        X_label, Y_label = X_Y_pairs[i]
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


# TODO: add state maybe?
app = dash.Dash()
app.layout = html.Div(
    [
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'Stable Diffusion v1.4', 'value': 'sd'},
                {'label': 'VQGAN + CLIP', 'value': 'vqgan'},
            ],
            value='sd'
        ),
        dcc.Input(id='text-input-a', type='text', placeholder='Enter Text A'),
        html.Div(id='output-a'),
        dcc.Input(id='text-input-b', type='text', placeholder='Enter Text B'),
        html.Div(id='output-b'),
        dcc.Dropdown(
            id='scatterplot-dropdown',
            options=options_menu,
            value='-1'
        ),
        dcc.Graph(id="scatterplots"),
        dcc.Dropdown(
            id='numberline-dropdown',
            options=options_menu,
            value='-1'
        ),
        dcc.Graph(id="number-line"),
    ]
)
@app.callback(
    Output('output-a', 'children'),
    Output('output-b', 'children'),
    Output("scatterplots", "figure"), 
    Output("number-line", "figure"), 
    Input('model-dropdown', 'value'),
    Input('scatterplot-dropdown', 'value'),
    Input('numberline-dropdown', 'value'),
    Input('text-input-a', 'value'),
    Input('text-input-b', 'value'),
)
def update_output(model_value, scatter_value, numberline_value, a_input, b_input):
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

    # NOTE: text_A change by input, fix text_B or not?
    # NOTE: might be difficult for users to come up with several phrases, give it by theme?
    # text_A = ["person to have intercourse with", "person to be intimate with", "person to have sex with", "person to kiss", "person to undress", "person to have coitus with"]
    # text_B = ["scientist", "researcher", "engineer", "physicist", "mathematician", "chemist"]
    # NOTE: but also one word works and might provide more interaction
    # text_A = ["person to have intercourse with"]
    # text_B = ["doctor"]
    if a_input is None or b_input is None:
        return "", "", scatter_fig, numberline_fig
    
    text_A = [a_input]
    text_B = [b_input]

    scatter_value, numberline_value = int(scatter_value), int(numberline_value)
    if scatter_value != -1: 
        X_label, Y_label = X_Y_pairs[scatter_value]
        cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, scatter_eat_score = process_inputs(text_A, text_B, f'{model_value}/{add_underscore(X_label)}', f'{model_value}/{add_underscore(Y_label)}')
        scatter_fig = scatterplot(scatter_fig, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y)
        scatter_fig.update_layout(title_text=f'Scatterplot, EAT Score: {scatter_eat_score}')
    if numberline_value != -1: 
        X_label, Y_label = X_Y_pairs[numberline_value]
        cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, numberline_eat_score = process_inputs(text_A, text_B, f'{model_value}/{add_underscore(X_label)}', f'{model_value}/{add_underscore(Y_label)}')
        numberline_fig = number_line(numberline_fig, X_label, Y_label, cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y)
        numberline_fig.update_layout(title_text=f'Number Line, EAT Score: {numberline_eat_score}')

    # display all pairs at once, for testing purposes initially    
    # scatter_data = []
    # for X_dir_name, Y_dir_name in X_Y_pairs:
    #     scatter_data.append(process_inputs(text_A, text_B, f'{model_value}/{add_underscore(X_dir_name)}', f'{model_value}/{add_underscore(Y_dir_name)}'))
    # scatter_fig = produce_scatterplot(scatter_data)
    # numberline_fig = produce_number_line(scatter_data)

    return f'A: {text_A}', f'B: {text_B}', scatter_fig, numberline_fig


if __name__ == "__main__":
    app.run_server(debug=True)