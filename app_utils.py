import os
import shutil
import base64
from dash import html
import math

def add_underscore(string_to_modify):
    return str.replace(string_to_modify, ' ', '_')

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

def generate_image_collage(dir_name):
    image_paths = os.listdir(dir_name)
    image_scatter = html.Div(
        [generate_image_element(f'{dir_name}/{image_path}') for image_path in image_paths],
        className='image-collage',
    )
    return image_scatter

def generate_image_element(image_path):
    with open(image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('ascii')
    image_element = html.Img(src=f"data:image/png;base64,{encoded_image}", style={'max-width': '100px', 'max-height': '100px'})
    return image_element
