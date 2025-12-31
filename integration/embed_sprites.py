# Generate and store embeddings in ChromaDB
from sprite_utils import load_model, get_embedding_from_file, load_image_paths, get_chroma_collection
import os
import numpy as np
from dotenv import load_dotenv
import tensorflow as tf
import pathlib

# Config
load_dotenv()
BATCH_SIZE = 32
CHROMA_BATCH_SIZE = 512
MODEL_PATH = './models/dungeon_model.keras'    # WHEREVER YOUR MODEL IS KEPT
DATASET_PATH = os.getenv("DATASET_PATH") # Path to dataset root directory
CHROMA_COLLECTION_NAME = 'sprite_embeddings'


# Load model

model, embedding_model = load_model(MODEL_PATH)

# Load dataset
image_paths = load_image_paths(DATASET_PATH)
print(f"Found {len(image_paths)} images for embedding.") # sanity check

# Generate embeddings
def load_and_process_image(path):         # Same as model training
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=4)
    image = tf.image.resize(image, [32, ])
    image = image / 255.0
    return image

dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(load_and_process_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

embeddings = embedding_model.predict(dataset)
print("Embedding shape is ", embeddings.shape) # Sanity check, correct shape

# Chromadb
collection = get_chroma_collection(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)

data_dir = pathlib.Path(DATASET_PATH)
def get_label_from_path(file_path):       # Same as model training
    path_object = pathlib.Path(file_path)
    relative_path = path_object.parent.relative_to(data_dir)
    return str(relative_path)

labels = [get_label_from_path(p) for p in image_paths]

# Using a for loop to batch load into chroma db to avoid the max batch size of like 5,000.
num_embeddings = len(embeddings)
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
