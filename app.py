import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load ResNet50 model (pre-trained on ImageNet)
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def run_similarity_search(query_image_path, image_directory, top_n=2, feature_file="image_features.pkl"):
    """
    Process query image, extract features, and find the most similar images.
    If features are not already saved, extract and save them from the database.

    Args:
        query_image_path (str): Path to the query image.
        image_directory (str): Path to the directory containing database images.
        top_n (int): Number of similar images to return.
        feature_file (str): Path to the saved feature file.

    Returns:
        List of top_n similar images with similarity scores.
    """
    def extract_features(img_path):
        """Extract feature vector from an image using ResNet50."""
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()

    def extract_features_from_directory(directory_path):
        """Extract features for all images in a directory."""
        feature_vectors = []
        file_paths = []
        for file_name in os.listdir(directory_path):
            if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(directory_path, file_name)
                features = extract_features(img_path)
                feature_vectors.append(features)
                file_paths.append(img_path)
        return feature_vectors, file_paths

    def save_features(feature_vectors, file_paths, output_file):
        """Save feature vectors to a file."""
        with open(output_file, 'wb') as f:
            pickle.dump((feature_vectors, file_paths), f)

    def load_features(input_file):
        """Load feature vectors from a file."""
        if os.path.exists(input_file):
            with open(input_file, 'rb') as f:
                return pickle.load(f)
        else:
            return None, None

    def find_similar_images(query_img_path, feature_vectors, file_paths, top_n):
        """Find the top N similar images to the query image."""
        query_features = extract_features(query_img_path)
        similarities = cosine_similarity([query_features], feature_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_n]
        similar_images = [(file_paths[i], similarities[i]) for i in top_indices]
        return similar_images

    def show_images(query_image_path, similar_images):
        """Visualize the query image and its similar results."""
        # Load the query image
        query_img = image.load_img(query_image_path, target_size=(224, 224))

        # Plot query image
        plt.figure(figsize=(12, 4))
        plt.subplot(1, len(similar_images) + 1, 1)
        plt.imshow(query_img)
        plt.title("Query Image")
        plt.axis("off")

        # Plot similar images
        for i, (img_path, similarity) in enumerate(similar_images):
            img = image.load_img(img_path, target_size=(224, 224))
            plt.subplot(1, len(similar_images) + 1, i + 2)
            plt.imshow(img)
            plt.title(f"Match {i+1}\nSimilarity: {similarity:.2f}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    # Step 1: Load or extract features from the image database
    feature_vectors, file_paths = load_features(feature_file)
    if feature_vectors is None or file_paths is None:
        print("Extracting features from the image database...")
        feature_vectors, file_paths = extract_features_from_directory(image_directory)
        save_features(feature_vectors, file_paths, feature_file)

    # Step 2: Search for similar images
    print(f"Querying similar images for: {query_image_path}")
    similar_images = find_similar_images(query_image_path, feature_vectors, file_paths, top_n)

    # Step 3: Display query image and similar images
    print("Top similar images:")
    for img_path, similarity in similar_images:
        print(f"  Image: {img_path}, Similarity: {similarity:.4f}")

    # Visualize results
    show_images(query_image_path, similar_images)

    return similar_images

# Example Usage
query_image_path = ".\FIND THIS MARBLE\onyx.jpg"
image_directory = ".\BACKUP MARBLE"
run_similarity_search(query_image_path, image_directory, top_n=2)
