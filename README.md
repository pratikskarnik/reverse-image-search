# Image Retrieval with EfficientNet and Annoy

## Overview

This Python script demonstrates content-based image retrieval using a pre-trained EfficientNet model and approximate nearest neighbor search implemented with Annoy. Given a query image, the script finds and displays the most visually similar images from a database of images.

## Features

- Utilizes a pre-trained EfficientNetB4 model for feature extraction.
- Implements approximate nearest neighbor search with Annoy for efficient retrieval.
- Supports customization of database images, query image, and retrieval parameters.
- Calculates and displays the Euclidean distances between query and similar images.
- Allows user interaction to view and close similar images.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

- Python (3.7 or higher)
- TensorFlow
- NumPy
- OpenCV (cv2)
- Annoy

You can install these dependencies using pip:

bash
pip install tensorflow numpy opencv-python annoy

## Usage
Clone or download this repository to your local machine.

Navigate to the project directory:


Copy code
```bash
cd image-retrieval-with-efficientnet
```
Edit the script to specify your database directory and query image path.

## Run the script:

Copy code
```bash
python image_retrieval.py
```
The script will display the top N similar images along with their distances.

To close the displayed images and exit, press any key.

## Customization
You can customize the script by:

Changing the pre-trained EfficientNet variant (e.g., EfficientNetB7).
Adjusting the num_trees parameter for the AnnoyIndex to trade off between speed and accuracy.
Modifying the input image paths (query_image_path) and the database directory (database_directory).
Setting the number of similar images to retrieve (top_n).

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The EfficientNet model is pre-trained on ImageNet and can be found in TensorFlow's Keras applications.
The Annoy library is used for efficient approximate nearest neighbor search.

## Author
Pratik Karnik

## Contact
For questions or suggestions, please feel free to contact pratikskarnik@gmail.com.
