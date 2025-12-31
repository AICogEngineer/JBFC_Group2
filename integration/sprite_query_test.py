import random
import os
from PIL import Image
import matplotlib.pyplot as plt
from sprite_utils import load_model, get_embedding_from_file, load_image_paths, get_chroma_collection

# Config
MODEL_PATH = './models/dungeon_model.keras'
DATASET_PATH = os.getenv("DATASET_PATH") # Path to dataset root directory
CHROMA_COLLECTION_NAME = 'sprite_embeddings'
CHROMA_DB_PATH = './chroma_db'

# Load model
model, embedding_model = load_model(MODEL_PATH)

# Load dataset
image_paths = load_image_paths(DATASET_PATH)

# Chromadb
collection = get_chroma_collection(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)

# grab random sprite
query_sprite = random.choice(image_paths)
query_vector = get_embedding_from_file(query_sprite, embedding_model)

# Query ChromaDB
results = collection.query(query_embeddings=[query_vector], n_results=5)

# Display results
fig, axes = plt.subplots(1, 6, figsize=(15, 5))
img = Image.open(query_sprite)
axes[0].imshow(img)
axes[0].set_title("Query Sprite")
axes[0].axis('off')

# show top 5 results
for i, res in enumerate(results['metadatas'][0]):
    img = Image.open(res['path'])
    axes[i+1].imshow(img)
    axes[i+1].set_title(res['label'])
    axes[i+1].axis('off')

plt.show()