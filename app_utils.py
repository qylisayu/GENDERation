import os
import shutil
import base64

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

