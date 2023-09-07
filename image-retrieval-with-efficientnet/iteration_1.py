# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 22:39:58 2023

@author: PRATIK
"""

import numpy as np
import tensorflow as tf
import cv2
import os
from heapq import nlargest

# Load a pre-trained ResNet model
resnet_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
model = tf.keras.Sequential([
    resnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
])

# Define a function to preprocess an image and extract features
def extract_features(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features

# Define a directory containing the database of images to search through
database_directory = "C:/Users/PRATIK/Pictures/Test"

# Extract features from all images in the database
database_features = {}
for filename in os.listdir(database_directory):
    if filename.endswith(".jpg"):
        image_path = os.path.join(database_directory, filename)
        features = extract_features(image_path)
        database_features[filename] = features

# Load the query image
query_image_path = "C:/Users/PRATIK/Pictures/aai.jpg"
query_features = extract_features(query_image_path)

# Define the number of similar images to retrieve
top_n = 10

# Initialize a list to store the top N similar images and their distances
top_similar_images = []
top_distances = [float('inf')] * top_n

# Calculate Euclidean distances between the query image and all database images
for filename, features in database_features.items():
    distance = np.linalg.norm(query_features - features)
    
    # Update the list of top similar images if the current image is closer than any in the top N
    if distance < max(top_distances):
        index_to_replace = top_distances.index(max(top_distances))
        top_similar_images.insert(index_to_replace, (filename, distance))
        top_similar_images = top_similar_images[:top_n]
        top_distances = [pair[1] for pair in top_similar_images]

# Display the top N most similar images and their distances
query_image = cv2.imread(query_image_path)
for i, (similar_image_filename, distance) in enumerate(top_similar_images):
    similar_image_path = os.path.join(database_directory, similar_image_filename)
    similar_image = cv2.imread(similar_image_path)
    print(f"Similar Image {i + 1}: Distance - {distance}")
    cv2.imshow(f"Similar Image {i + 1}", similar_image)
    

cv2.waitKey(0)
cv2.destroyAllWindows()
