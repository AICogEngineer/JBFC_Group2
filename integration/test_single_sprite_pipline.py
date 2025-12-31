import os
import numpy as np
import pathlib
import tensorflow as tf
from dotenv import load_dotenv
from sprite_utils import (
    load_model,
    get_embedding_from_file,
    get_chroma_collection
)
from embed_sprites import normalize_vec
from PIL import Image

# Config
load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")
data_dir = pathlib.Path(DATASET_PATH)
MODEL_PATH = "./models/dungeon_model.keras"
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "sprite_embeddings"
TEST_DATASET_PATH = os.getenv("TEST_IMAGE_PATH")
TEST_SPRITE_PATH = os.path.join(TEST_DATASET_PATH, "img_25.png")

TOP_K_CLASSES = 9
TOP_K_NEIGHBORS = 5

# Load model
model, embedding_model = load_model(MODEL_PATH)

# Load all class labels used in training
# Include all subfolders as labels, exactly like during training
class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

# Preprocess image
def load_image_for_model(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((32, 32))
    img_array = np.array(img, dtype=np.float32)
    return tf.expand_dims(img_array, axis=0)

img_tensor = load_image_for_model(TEST_SPRITE_PATH)

# Classifier prediction
probs = model.predict(img_tensor)[0]

# Get top-K predictions
top_indices = np.argsort(probs)[-TOP_K_CLASSES:][::-1]
candidate_labels = [class_names[i] for i in top_indices]

print("\n=== CLASSIFIER OUTPUT ===")
for i in top_indices:
    print(f"{class_names[i]}: {probs[i]:.3f}")

print("\nCandidate labels:", candidate_labels)

# Generate embedding
query_vec = get_embedding_from_file(TEST_SPRITE_PATH, embedding_model)
query_vec = normalize_vec(query_vec)

# Chromadb query
collection = get_chroma_collection(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)

# Get more neighbors than needed so we can filter
results = collection.query(query_embeddings=[query_vec], n_results=TOP_K_NEIGHBORS*5)

top_classes_set = set(candidate_labels)
filtered_neighbors = []

# Filter neighbors by comparing the first 1-2 folder levels (matches classifier labels)
for meta in results["metadatas"][0]:
    # meta['label'] is the stored relative path like 'dungeon/statues/statue_01.png'
    for candidate in top_classes_set:
        if meta["label"].startswith(candidate):
            filtered_neighbors.append(meta)
            break
    if len(filtered_neighbors) >= TOP_K_NEIGHBORS:
        break

# Display results
print("\n=== EMBEDDING NEIGHBORS (filtered) ===")
for i, meta in enumerate(filtered_neighbors):
    print(f"{i+1}. {meta['label']} → {meta['path']}")
