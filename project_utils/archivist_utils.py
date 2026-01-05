import os
import pathlib
import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from PIL import Image
from chromadb import PersistentClient
import numpy as np
from chromadb.utils import embedding_functions
from PIL import Image

# Config
BATCH_SIZE = 32
CHROMA_BATCH_SIZE = 512
CHROMA_COLLECTION_NAME = 'sprite_embeddings'
CHROMA_DB_PATH = 'chroma_db'

# Load CNN and embedding layer
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    embedding_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("embeddings").output
    )
    return model, embedding_model

#Return flattened embedding vector for a single image
def get_embedding_from_file(path, embedding_model):
    img = Image.open(path).convert("RGBA")
    img = img.resize((32, 32))
    img = np.array(img)
    img = tf.expand_dims(img, axis=0)
    embedding = embedding_model(img, training=False)
    return embedding.numpy().flatten()

def load_image_paths(dataset_path):
    data_dir = pathlib.Path(dataset_path)
    return [str(p) for p in data_dir.rglob('*.png')]

def get_chroma_collection(db_path, collection_name):
    client = PersistentClient(db_path)
    return client.get_or_create_collection(name=collection_name, embedding_function=embedding_functions.DefaultEmbeddingFunction(), metadata={"distance_metric": "cosine"})

def normalize_vec(vec):
    return vec / norm(vec)

def get_label_from_path(file_path, data_dir):
    path_object = pathlib.Path(file_path)
    return str(path_object.parent.relative_to(data_dir))

def load_and_process_image(path):
    def _process(p):
        img = Image.open(p.numpy().decode("utf-8")).convert("RGBA")
        img = img.resize((32, 32))
        return np.array(img, dtype=np.float32)

    image = tf.py_function(_process, [path], tf.float32)
    image.set_shape([32, 32, 4])
    return image

def load_sprite_embeddings_into_chromadb(model_path, dataset_path):
    model, embedding_model = load_model(model_path)
    image_paths = load_image_paths(dataset_path)
    print(f"Found {len(image_paths)} images.")

    data_dir = pathlib.Path(dataset_path)
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_process_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    embeddings = embedding_model.predict(dataset)
    print("Embedding shape:", embeddings.shape)

    collection = get_chroma_collection(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)
    
    # Clear existing data
    print("Clearing existing collection...")
    try:
        existing_ids = collection.get()['ids']
        if existing_ids:
            collection.delete(existing_ids)
            print(f"Deleted {len(existing_ids)} existing items.")
    except Exception as e:
        print(f"Error clearing collection: {e}")

    labels = [get_label_from_path(p, data_dir) for p in image_paths]

    # Batch insert into ChromaDB
    num_embeddings = len(embeddings)
    for start_idx in range(0, num_embeddings, CHROMA_BATCH_SIZE):
        end_idx = start_idx + CHROMA_BATCH_SIZE
        batch_embeddings = embeddings[start_idx:end_idx]
        batch_embeddings = np.array([normalize_vec(v) for v in batch_embeddings])
        batch_labels = labels[start_idx:end_idx]
        batch_paths = image_paths[start_idx:end_idx]

        collection.add(
            embeddings=batch_embeddings.tolist(),
            metadatas=[{"label": batch_labels[i], "path": batch_paths[i]} for i in range(len(batch_embeddings))],
            ids=[f"sprite_{start_idx+i}" for i in range(len(batch_embeddings))]
        )
        print(f"Inserted {min(end_idx, num_embeddings)}/{num_embeddings}")

    print("ChromaDB ingestion complete.")
    print("Total vectors stored:", collection.count())
