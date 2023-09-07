import numpy as np
import tensorflow as tf
import cv2
import os
from annoy import AnnoyIndex

# Load a pre-trained ResNet model
resnet_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
model = tf.keras.Sequential([
    resnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
])

# Define a function to preprocess an image and extract features
def extract_features(image):
    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features

# Define a directory containing the database of images to search through
database_directory = "C:/Users/PRATIK/Pictures/Test"

# Extract features from all images in the database and cache them
database_features = []
database_filenames = []
for filename in os.listdir(database_directory):
    if filename.endswith(".jpg"):
        image_path = os.path.join(database_directory, filename)
        features = extract_features(image_path)
        database_features.append(features.flatten())  # Flatten the features
        database_filenames.append(filename)

# Create an AnnoyIndex for approximate nearest neighbor search
num_trees = 100  # You can adjust this parameter for trade-off between speed and accuracy
feature_dim = len(database_features[0])
annoy_index = AnnoyIndex(feature_dim, 'euclidean')

# Add database features to the AnnoyIndex
for i, feature in enumerate(database_features):
    annoy_index.add_item(i, feature)

# Build the index
annoy_index.build(num_trees)

# Load the query image
query_image_path = "C:/Users/PRATIK/Pictures/pratik3.jpg"
query_features = extract_features(query_image_path).flatten()  # Flatten the query features

# Define the number of similar images to retrieve
top_n = 10

# Find the top N similar images
similar_indices = annoy_index.get_nns_by_vector(query_features, top_n)

# Display the top N most similar images and their distances
query_image = cv2.imread(query_image_path)
for i, index in enumerate(similar_indices):
    similar_image_filename = database_filenames[index]
    similar_image_path = os.path.join(database_directory, similar_image_filename)
    similar_image = cv2.imread(similar_image_path)
    # Calculate the distance manually
    distance = np.linalg.norm(query_features - database_features[index])
    print(f"Similar Image {i + 1}: Distance - {distance}")
    cv2.imshow(f"Similar Image {i + 1}", similar_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
