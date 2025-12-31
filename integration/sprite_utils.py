import os
import pathlib
import tensorflow as tf
from chromadb import PersistentClient
import numpy as np
from chromadb.utils import embedding_functions
from PIL import Image

# Config
IMG_HEIGHT = 32
IMG_WIDTH = 32

# Load CNN and embedding layer
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    embedding_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("embedding").output
    )
    return model, embedding_model

#Return flattened embedding vector for a single image
def get_embedding_from_file(path, embedding_model):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
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