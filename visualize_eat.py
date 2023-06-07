from PIL import Image
import numpy as np
import torch
from CLIP import clip
import os
import matplotlib.pyplot as plt


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
        plt.plot(num, 0, marker='o', markersize=8, color=color, label=label)


def scale(x, new_min=-1, new_max=1, old_min=-2, old_max=2):
    return ((new_max - new_min) * (x - old_min)) / (old_max - old_min) + new_min


# calculate EAT score
def calculate_eat_score(X, Y, A, B):
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

    # Calculate association scores for elements in the union of sets X and Y
    union_set = np.concatenate((X, Y))
    association_scores_union = np.array([s(w, A, B) for w in union_set])[:, 0]
    std_dev_association = np.std(association_scores_union)

    # Calculate EAT score
    eat_score = (mean_association_x - mean_association_y) / std_dev_association
    return cos_a_scores_x, cos_b_scores_x, cos_a_scores_y, cos_b_scores_y, eat_score


# calculate EAT score (with plots)
def calculate_and_visualize_eat(X, Y, A, B):
    # Calculate association scores for elements in set X
    similarity_scores_x = np.array([s(x, A, B) for x in X])
    association_scores_x = similarity_scores_x[:, 0]
    mean_association_x = np.mean(association_scores_x)

    # Calculate association scores for elements in set Y
    similarity_scores_y = np.array([s(y, A, B) for y in Y])
    association_scores_y = similarity_scores_y[:, 0]
    mean_association_y = np.mean(association_scores_y)

    # Number line plotting the cosine similarities
    plt.axhline(0, color='black')  # Draw the number line at y=0 (for individual points)
    plt.axhline(1, color='black')  # Draw the number line at y=1 (for mean)
    cos_a_scores_x = similarity_scores_x[:, 1]
    cos_b_scores_x = similarity_scores_x[:, 2]
    visualization_scores_x = [scale((-cos_a_scores_x[i] + cos_b_scores_x[i])) for i in range(len(cos_a_scores_x))]
    # print(visualization_scores_x)
    mean_visualization_x = np.mean(visualization_scores_x)
    plot_dots(visualization_scores_x, 'blue')
    plt.plot(mean_visualization_x, 1, marker='x', markersize=8, color='blue')

    cos_a_scores_y = similarity_scores_y[:, 1]
    cos_b_scores_y = similarity_scores_y[:, 2]
    visualization_scores_y = [scale((-cos_a_scores_y[i] + cos_b_scores_y[i])) for i in range(len(cos_a_scores_y))]
    # print(visualization_scores_y)
    mean_visualization_y = np.mean(visualization_scores_y)
    plot_dots(visualization_scores_y, 'red')
    plt.plot(mean_visualization_y, 1, marker='x', markersize=8, color='red')
    
    plt.yticks([])  # Remove y-axis ticks
    plt.savefig('number-line.png')
    plt.clf()

    # Scatter plotting the cosine similarities
    plt.scatter(cos_a_scores_x, cos_b_scores_x, color='blue', label='X', alpha=0.1)
    mean_x = np.mean(cos_a_scores_x), np.mean(cos_b_scores_x)
    plt.plot(mean_x[0], mean_x[1], marker='x', markersize=8, color='blue')
    plt.scatter(cos_a_scores_y, cos_b_scores_y, color='red', label='Y', alpha=0.1)
    mean_y = np.mean(cos_a_scores_y), np.mean(cos_b_scores_y)
    plt.plot(mean_y[0], mean_y[1], marker='x', markersize=8, color='red')
    plt.xlabel('A Similarity Scores')
    plt.ylabel('B Similarity Scores')
    plt.legend()
    plt.axis('equal')
    plt.savefig('scatter-plot.png')

    # Calculate association scores for elements in the union of sets X and Y
    union_set = np.concatenate((X, Y))
    association_scores_union = np.array([s(w, A, B) for w in union_set])[:, 0]
    std_dev_association = np.std(association_scores_union)

    # Calculate EAT score
    eat_score = (mean_association_x - mean_association_y) / std_dev_association
    return eat_score


if __name__ == "__main__": 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')

    # load model and image preprocessing
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    text_A = ["person to have intercourse with", "person to be intimate with", "person to have sex with", "person to kiss", "person to undress", "person to have coitus with"] # same as https://arxiv.org/pdf/2212.11261.pdf
    text_B = ["scientist", "researcher", "engineer", "physicist", "mathematician", "chemist"] # same as https://arxiv.org/pdf/2212.11261.pdf
    X_dir_name = "sd/photo_portrait_of_a_female_doctor"
    Y_dir_name = "sd/photo_portrait_of_a_male_doctor"

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

    # TODO: not the most clean code lol
    print(f'EAT score: {calculate_eat_score(X, Y, A, B)[4]}')
    print(f'EAT score: {calculate_and_visualize_eat(X, Y, A, B)}')