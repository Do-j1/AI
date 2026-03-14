import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import random

def visualize_letter_landmarks_normalized(all_letters_landmarks, use_tsne=False, perplexity = 10):
    """
    Visualize normalized hand landmarks for multiple letters.

    Parameters:
    - all_letters_landmarks: dict
        { 'letter1': [img1_landmarks, img2_landmarks, ...], ... }
        each img_landmarks: np.array shape (21, 3), already normalized
    - use_tsne: bool
        If True, use t-SNE; else PCA.
    - perplexity: int
        t-SNE perplexity, only used if use_tsne=True
    """
    
    flattened_data = []
    labels = []

    # Flatten and collect data
    for letter, images in all_letters_landmarks.items():
        for img in images:
            flattened_data.append(img.flatten())  # 21*3 = 63
            labels.append(letter)
    
    flattened_data = np.array(flattened_data)
    labels = np.array(labels)
    
    # Reduce dimensions
    if use_tsne:
        reducer = TSNE(n_components=2, perplexity =perplexity, random_state=42)
        data_2d = reducer.fit_transform(flattened_data)
        title = f"t-SNE of Hand Landmarks (perplexity={perplexity})"
    else:
        reducer = PCA(n_components=2)
        data_2d = reducer.fit_transform(flattened_data)
        title = "PCA of Hand Landmarks"

    # Plot
    plt.figure(figsize=(10,7))
    for letter in np.unique(labels):
        idx = labels == letter
        plt.scatter(data_2d[idx,0], data_2d[idx,1], label=letter, alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()
all_letters_landmarks = np.load("all_letters_landmarks.npy", allow_pickle=True).item()

total_images = sum(len(images) for images in all_letters_landmarks.values())
print(f"Total number of images: {total_images}")

subset_letters_landmarks = {}
sample_per_letter = 100  # adjust for speed

for letter, images in all_letters_landmarks.items():
    images_list = list(images)  # <-- convert to a list
    if len(images_list) > sample_per_letter:
        subset_letters_landmarks[letter] = random.sample(images_list, sample_per_letter)
    else:
        subset_letters_landmarks[letter] = images_list

visualize_letter_landmarks_normalized(subset_letters_landmarks, use_tsne=True, perplexity=150)

#visualize_letter_landmarks_normalized(all_letters_landmarks , False)


