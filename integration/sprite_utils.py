import os
import pathlib
import tensorflow as tf
from chromadb import PersistentClient
import numpy as np

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
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=4)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    embedding = embedding_model(img)
    return embedding.numpy().flatten()

def load_image_paths(dataset_path):
    data_dir = pathlib.Path(dataset_path)
    return [str(p) for p in data_dir.rglob('*.png')]

def get_chroma_collection(db_path, collection_name):
    client = PersistentClient(db_path)
    return client.get_or_create_collection(collection_name)