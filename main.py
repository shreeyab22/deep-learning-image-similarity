import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg
from tqdm import tqdm
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# STEP 0: Optional - Combine CNN + Color Histogram
# ==============================
def color_histogram(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0,256,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

# ==============================
# STEP 1: Load Pre-trained Model
# ==============================
print("Loading MobileNetV2 model...")
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)
print("Model loaded successfully!")

# ==============================
# STEP 2: Feature Extraction
# ==============================
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    features = model.predict(img, verbose=0)
    return features.flatten()

# ==============================
# STEP 3: Load Dataset Images
# ==============================
dataset_path = "images_compressed"

image_paths = []
features_list = []

print("Extracting features from dataset...")

for img_name in tqdm(os.listdir(dataset_path)):
    img_path = os.path.join(dataset_path, img_name)
    feature = extract_features(img_path)
    if feature is not None:
        image_paths.append(img_path)
        features_list.append(feature)

features_list = np.array(features_list)
print("Feature extraction completed!")

# ==============================
# STEP 4: Similarity Search (MobileNet + optional color histogram)
# ==============================
def find_similar_images(query_image, top_k=5, use_color_histogram=False):
    query_feature = extract_features(query_image)
    
    # Optional: combine with color histogram
    if use_color_histogram:
        query_color = color_histogram(query_image)
        query_feature = np.concatenate([query_feature, query_color])
    
    dataset_features_combined = []
    for f, path in zip(features_list, image_paths):
        if use_color_histogram:
            c_feat = color_histogram(path)
            combined = np.concatenate([f, c_feat])
            dataset_features_combined.append(combined)
        else:
            dataset_features_combined.append(f)
    
    dataset_features_combined = np.array(dataset_features_combined)
    
    # Compute cosine similarity
    similarities = cosine_similarity([query_feature], dataset_features_combined)[0]
    
    # Get top indices
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Get top image paths and scores
    top_images = [image_paths[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]
    
    return top_images, top_scores

# ==============================
# STEP 5: Query Image
# ==============================
query_image_path = "query/test.jpg"
top_k = 5  # You can change this to show more images
use_color_histogram = True  # Set to True to combine MobileNet + color histogram

similar_images, similarity_scores = find_similar_images(
    query_image_path, top_k=top_k, use_color_histogram=use_color_histogram
)

# ==============================
# STEP 6: Plot Query + Top Similar Images
# ==============================
plt.figure(figsize=(18,5))

# Show query image
plt.subplot(1, len(similar_images)+1, 1)
img = mpimg.imread(query_image_path)
plt.imshow(img)
plt.title("Query Image")
plt.axis('off')

# Show top similar images with scores
for i, img_path in enumerate(similar_images):
    plt.subplot(1, len(similar_images)+1, i+2)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(f"Score: {similarity_scores[i]:.2f}")
    plt.axis('off')

# Save the plot automatically
plt.savefig("similarity_results.png")
plt.show()

print("Query and top similar images plotted and saved as 'similarity_results.png'.")

def get_similar_images(query_image_path, top_k=5):
    return find_similar_images(query_image_path, top_k)
