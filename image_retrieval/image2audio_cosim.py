import numpy as np
import cv2
import os
from playsound import playsound

dataset_dir = '$$$'
new_image_path = '$$$' 

# 组织：label--img1，img2...

def load_dataset(image_dir):
    images = []
    labels = []
    for label in range(10):
        class_dir = os.path.join(image_dir, str(label))
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100))
            images.append(img.flatten())
            labels.append(label)
    return np.array(images), np.array(labels)

def load_new_image(new_image_path):
    img = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    return img.flatten()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

images, labels = load_dataset(dataset_dir)
new_image = load_new_image(new_image_path)
similarities = [cosine_similarity(new_image, img) for img in images]

most_similar_index = np.argmax(similarities)
predicted_label = labels[most_similar_index]
playsound(f"audio{predicted_label}.mp3")
