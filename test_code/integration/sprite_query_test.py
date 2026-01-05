import random
import os
from PIL import Image
import matplotlib.pyplot as plt
from sprite_utils import load_model, get_embedding_from_file, load_image_paths, get_chroma_collection
from dotenv import load_dotenv
from numpy.linalg import norm
from embed_sprites import normalize_vec
from pathlib import Path

# Config
MODEL_PATH = './models/dungeon_model_ab_og.keras'
DATASET_PATH = os.getenv("DATASET_PATH") # Path to dataset root directory
CHROMA_COLLECTION_NAME = 'sprite_embeddings'
CHROMA_DB_PATH = './chroma_db'
load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH") # Path to dataset root directory
SAVE_DIR = "test_queries"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model
model, embedding_model = load_model(MODEL_PATH)

# Load dataset
image_paths = load_image_paths(DATASET_PATH)

# Chromadb
collection = get_chroma_collection(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)


# grab random sprite
query_sprite = random.choice(image_paths)
query_vector = get_embedding_from_file(query_sprite, embedding_model)



# Get label function for query results
def get_label_from_path(file_path, dataset_root):
    path_obj = Path(file_path)
    relative_path = path_obj.parent.relative_to(dataset_root)
    return str(relative_path)

# Get query sprite label
query_label = get_label_from_path(query_sprite, DATASET_PATH)

# Normalize query vector
query_vector = normalize_vec(query_vector)

# Query ChromaDB
results = collection.query(query_embeddings=[query_vector], n_results=5)

# Save results
existing_files = os.listdir(SAVE_DIR)
counter = len(existing_files)
prefix = f"query_{counter:03d}"

# Display results
fig, axes = plt.subplots(1, 6, figsize=(15, 5))
img = Image.open(query_sprite)
axes[0].imshow(img)
axes[0].set_title(f"Query Sprite\n{query_label}", fontsize=8)
axes[0].axis('off')

# show/save top 5 results
for i, res in enumerate(results['metadatas'][0]):
    img = Image.open(res['path'])
    axes[i+1].imshow(img)
    label = res['label']
    axes[i+1].set_title(f"Top {i+1}\n{label}", fontsize=8)
    axes[i+1].axis('off')

plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_figure.png"))
plt.close(fig)
