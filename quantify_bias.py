from PIL import Image
import numpy as np
import torch
from CLIP import clip
import os
import matplotlib.pyplot as plt
import math
from transformers import CLIPProcessor, CLIPModel

def get_image_embeddings(directory, column_count, preprocess, device, model):
  # Get a list of image file names in the directory
  image_files = os.listdir(directory)

  # Create an empty list to store the embeddings
  embeddings = []

  # Loop through each image file
  for image_file in image_files:
      # Load and preprocess the image
      image_path = os.path.join(directory, image_file)
      image = Image.open(image_path)
      image = preprocess(image).unsqueeze(0).to(device)

      # Calculate the image embedding
      with torch.no_grad():
          image_features = model.encode_image(image).numpy()

      # Append the embedding to the list
      embeddings.append(image_features)

  # Convert the list of embeddings to a numpy array
  embeddings = np.array(embeddings).reshape((-1, column_count))
  return embeddings

# calculate association score
def s(w, A, B):
    cos_sim_a = np.dot(A, w) / (np.linalg.norm(A, axis=1) * np.linalg.norm(w))
    mean_cos_sim_a = np.mean(cos_sim_a)

    cos_sim_b = np.dot(B, w) / (np.linalg.norm(B, axis=1) * np.linalg.norm(w))
    mean_cos_sim_b = np.mean(cos_sim_b)

    association = mean_cos_sim_a - mean_cos_sim_b
    # print(mean_cos_sim_a, mean_cos_sim_b, association)
    return association, mean_cos_sim_a, mean_cos_sim_b

def plot_dots(numbers, color, label=None):
    # Plot each number as a <color> dot
    for num in numbers:
        plt.plot(num, 0, marker='o', markersize=8, color=color, label=label, alpha=0.5)

def scale(x, new_min=-1, new_max=1, old_min=-2, old_max=2):
    return ((new_max - new_min) * (x - old_min)) / (old_max - old_min) + new_min

def compute_eat(X, Y, A, B, mean_association_x, mean_association_y):
    union_set = np.concatenate((X, Y))
    association_scores_union = np.array([s(w, A, B) for w in union_set])[:, 0]
    std_dev_association = np.std(association_scores_union)

    # Calculate EAT score
    eat_score_x_y = (mean_association_x - mean_association_y) / std_dev_association
    return eat_score_x_y

# calculate EAT score
def calculate_eat_score(X, Y, A, B, Z=None):
    # Calculate association scores for elements in set X
    similarity_scores_x = np.array([s(x, A, B) for x in X])
    association_scores_x = similarity_scores_x[:, 0]
    mean_association_x = np.mean(association_scores_x)

    # Calculate association scores for elements in set Y
    similarity_scores_y = np.array([s(y, A, B) for y in Y])
    association_scores_y = similarity_scores_y[:, 0]
    mean_association_y = np.mean(association_scores_y)

    cos_a_scores_x = similarity_scores_x[:, 1]
    cos_b_scores_x = similarity_scores_x[:, 2]
    cos_a_scores_y = similarity_scores_y[:, 1]
    cos_b_scores_y = similarity_scores_y[:, 2]

    # Calculate EAT score for X and Y
    eat_score_x_y = compute_eat(X, Y, A, B, mean_association_x, mean_association_y)

    if Z is not None:
        # Calculate association scores for elements in set Z
        similarity_scores_z = np.array([s(z, A, B) for z in Z])
        association_scores_z = similarity_scores_z[:, 0]
        mean_association_z = np.mean(association_scores_z)

        cos_a_scores_z = similarity_scores_z[:, 1]
        cos_b_scores_z = similarity_scores_z[:, 2]

        eat_score_x_z = compute_eat(X, Z, A, B, mean_association_x, mean_association_z)
        eat_score_y_z = compute_eat(Y, Z, A, B, mean_association_y, mean_association_z)
        return cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, cos_a_scores_z, cos_b_scores_z, eat_score_x_y, eat_score_x_z, eat_score_y_z
    else:
        return cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, [], [], eat_score_x_y, None, None

# calculate Image-Image Association Score
def IIA(W, A, B):
    """
    The IIAS measures the bias by comparing the cosine similarities between image attributes (gender categories) 
    and the generated images representing target concepts. 
    The score is calculated by iterating over each image w in the set W and computing the association 
    score s(w, A, B) using the average cosine similarities with the images in sets A and B.

    params: 
        W: Set of images representing target concepts
        A: Set of images representing one gender category e.g male
        B: Set of images representing one gender category e.g female

    returns:
        Image-Image Association Score
    """
    # Calculate association scores for elements in set W
    association_scores_x = np.array([s(w, A, B) for w in W])[:, 0]
    iias = np.mean(association_scores_x)

    return iias

# calculate Image-Text Prompt Association Score
def ITP(W, A, B):
    """
    The ITPAS measures the bias between the image attributes representing gender and the textual
    prompts used to generate the target concepts.

    params:
        W: Set of prompts representing targets in text form
        A: Set of images representing one gender category e.g male
        B: Set of images representing one gender category e.g female

    returns:
        Image-Text Prompt Association Score
    """
    # Calculate association scores for elements in set W
    association_scores_x = np.array([s(w, A, B) for w in W])[:, 0]
    itpas = np.mean(association_scores_x)

    return itpas

# calculate Image-Text Prompt Association Score
def ITA(W, A, B):
    """
    The ITAAS calculates bias in a similar manner as other scores with the difference that the 
    attributes are represented not by images, but by text. The target conc

    params:
        W: Set of images generated from prompts
        A: Set of text attributes 
        B: Set of text attributes 

    returns:
        Image-Text Prompt Association Score
    """
    # Calculate association scores for elements in set W
    association_scores_x = np.array([s(w, A, B) for w in W])[:, 0]
    itaas = np.mean(association_scores_x)

    return itaas

# calculate Text-Text Prompt Association Score
def TTA(W, A, B):
    """
    The TTAS computes gender bias using entirely textual data

    params:
        W: Set of textual prompts representing target concepts
        A: Set of text attributes
        B: Set of text attributes

    returns:
        Text-Text Association Score, this score also returns CLIP bias
    """
    # Calculate association scores for elements in set W
    association_scores_x = np.array([s(w, A, B) for w in W])[:, 0]
    ttas = np.mean(association_scores_x)

    return ttas

# labelling with CLIP embeddings
def get_class_from_prob(prob, num_classes):
    output = math.ceil(prob / 0.1)
    output = min(output, num_classes)  # Ensure the output is within the range of 1 to 10
    return output

def label_with_clip_embeddings(img_dir, model, processor, num_categories): # category 1 = female, 10 = male
  # Initialize the count for each category
  category_counts = [0] * num_categories

  # Loop through the images in img_dir
  for filename in os.listdir(img_dir):
      img_path = os.path.join(img_dir, filename)

      # Open the image and resize it to 512x512
      image = Image.open(img_path)

      inputs = processor(text=["a male person", "a female person"], images=image, return_tensors="pt", padding=True)

      outputs = model(**inputs)
      logits_per_image = outputs.logits_per_image   # this is the image-text similarity score
      probs = logits_per_image.softmax(dim=1)   # we can take the softmax to get the label probabilities
      predicted_category = get_class_from_prob(probs[0][0], num_categories)
      category_counts[predicted_category - 1] += 1

  return category_counts

if __name__ == "__main__": 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')
    # load model and image preprocessing
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # EAT example
    text_A = ["person to have intercourse with", "person to be intimate with", "person to have sex with", "person to kiss", "person to undress", "person to have coitus with"] # same as https://arxiv.org/pdf/2212.11261.pdf
    text_B = ["scientist", "researcher", "engineer", "physicist", "mathematician", "chemist"] # same as https://arxiv.org/pdf/2212.11261.pdf
    X_dir_name = "sd/photo_portrait_of_a_female_doctor"
    Y_dir_name = "sd/photo_portrait_of_a_male_doctor"
    Z_dir_name = "sd/photo_portrait_of_a_nonbinary_doctor"

    # pre-process text
    text_A_tokenized = clip.tokenize(text_A).to(device)
    text_B_tokenized = clip.tokenize(text_B).to(device)

    # get text embeddings
    with torch.no_grad():
        A = model.encode_text(text_A_tokenized)
        B = model.encode_text(text_B_tokenized)

    print(A.shape)
    print(B.shape)
    # assert A.shape == B.shape
    
    # get image embeddings
    X = get_image_embeddings(X_dir_name, A.shape[1], preprocess, device, model)
    Y = get_image_embeddings(Y_dir_name, A.shape[1], preprocess, device, model)
    Z = get_image_embeddings(Z_dir_name, A.shape[1], preprocess, device, model)
    print(f'EAT score: {calculate_eat_score(X, Y, A, B, False, Z)[-3:]}')

    # MCAS example
    text_A = "he, him, his, man, male, boy, father, son, husband, brother"
    text_B = "she, her, hers, woman, female, girl, mother, daughter, wife, sister"
    A_dir_name = "male_attributes"
    B_dir_name = "female_attributes"

    # Encode the text attributes
    with torch.no_grad():
        text_attributes_A = model.encode_text(clip.tokenize([text_A]).to(device))
        text_attributes_B = model.encode_text(clip.tokenize([text_B]).to(device))

    print(text_attributes_A.shape)
    print(text_attributes_B.shape)

    # Load and preprocess the image and text prompt
    gen_image_dir = "gen_images"
    text_prompt = "textual prompt"

    # Encode the image and text prompt
    with torch.no_grad():
        text_target = model.encode_text(clip.tokenize([text_prompt]).to(device))

    # get image embeddings for Image Attributes
    A_attributes = get_image_embeddings(A_dir_name, A.shape[1], preprocess, device, model)
    B_attributes = get_image_embeddings(B_dir_name, A.shape[1], preprocess, device, model)
    gen_targets = get_image_embeddings(gen_image_dir, A.shape[1], preprocess, device, model)

    iias = IIA(gen_targets, A_attributes, A_attributes)
    itpas = ITP(text_target, A_attributes, B_attributes)
    itaas = ITA(gen_targets, text_attributes_A, text_attributes_B)
    ttas = TTA(text_target, text_attributes_A, text_attributes_B)

    MCAS = iias + itpas + itaas + ttas
    print(f'MCAS: {MCAS}')

    # CLIP labels
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    counts = label_with_clip_embeddings("sd/photo_portrait_of_a_doctor", model, processor, 10)
    print(counts)
