# Generate and store embeddings in ChromaDB

import os
import pathlib
import random 
import tensorflow as tf
import chromadb
from chromadb.config import Settings
import numpy as np

# Config has to match training model-----------------------
IMG_HEIGHT = 32
IMG_WIDTH = 32
BATCH_SIZE = 32

DATA_DIR = './data/Dungeon Crawl Stone Soup Full_v2/'
MODEL_PATH = './models/dungeon_model.keras'    # WHEREVER YOUR MODEL IS KEPT
CHROMA_COLLECTION_NAME = 'sprite_embeddings'
# ----------------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)   # WHEREVER YOUR MODEL IS KEPT

embedding_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer("embedding").output
)

data_dir = pathlib.Path(DATA_DIR)
image_paths = list(data_dir.rglob('*.png'))
image_paths = [str(p) for p in image_paths]

print(f"Found {len(image_paths)} images for embedding.") # sanity check

def get_label_from_path(file_path):       # Same as model training
    path_object = pathlib.Path(file_path)
    relative_path = path_object.parent.relative_to(data_dir)
    return str(relative_path)

labels = [get_label_from_path(p) for p in image_paths]

def load_and_process_image(path):         # Same as model training
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0
    return image

dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(load_and_process_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

embeddings = embedding_model.predict(dataset)
print("Embedding shape is ", embeddings.shape) # Sanity check, correct shape


# initialize chromadb
client = chromadb.PersistentClient("./chroma_db")   # saves to chroma_db directory

collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME    # "sprite_embeddings"
)

CHROMA_BATCH_SIZE = 512 # "Safe standard" size chunks for chroma batching
num_embeddings = len(embeddings) # 6024

# Using a for loop to batch load into chroma db to avoid the max batch size of like 5,000.
for start_idx in range(0, num_embeddings, CHROMA_BATCH_SIZE):
    end_idx = start_idx + CHROMA_BATCH_SIZE

    batch_embeddings = embeddings[start_idx:end_idx]
    batch_labels = labels[start_idx:end_idx]
    batch_paths = image_paths[start_idx:end_idx]

    collection.add(
        embeddings=batch_embeddings.tolist(),
        metadatas=[
            {
                "label": batch_labels[i],
                "path": batch_paths[i]
            }
            for i in range(len(batch_embeddings))
        ],
        ids=[
            f"sprite_{start_idx + i}"
            for i in range(len(batch_embeddings))
        ]
    )
    print(f"Inserted {end_idx if end_idx < num_embeddings else num_embeddings}/{num_embeddings}")

print("ChromaDB ingestion complete.")
print("Total vectors stored:", collection.count())
