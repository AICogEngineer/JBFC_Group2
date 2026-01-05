import os
import sys
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
MODEL_PATH = "models/dungeon_model_ab_og.keras"

# Check for argument
if len(sys.argv) < 2:
    print("Error: No image path provided.")
    sys.exit(1)

TEST_SPRITE_PATH = sys.argv[1] 

TOP_K_CLASSES = 9

# Load model
model, embedding_model = load_model(MODEL_PATH)

# Load class labels
class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

# Preprocess image
def load_image_for_model(path):
    try:
        img = Image.open(path).convert("RGBA")
        img = img.resize((32, 32))
        img_array = np.array(img, dtype=np.float32)
        return tf.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

img_tensor = load_image_for_model(TEST_SPRITE_PATH)

# Classifier prediction
probs = model.predict(img_tensor, verbose=0)[0] # verbose=0 silences Keras loading bars

# Get top prediction
top_indices = np.argsort(probs)[-TOP_K_CLASSES:][::-1]
candidate_labels = [class_names[i] for i in top_indices]

# <--- CHANGE 3: Only print the final winner (The Top 1 Label)
# We comment out the debug prints so the automation script sees only the folder name
# print("=== CLASSIFIER OUTPUT ===") ...

# The first item in candidate_labels is the highest probability match
best_label = candidate_labels[0]
print(best_label)