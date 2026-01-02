from sprite_utils import load_model, get_chroma_collection, load_image_paths
<<<<<<< Updated upstream
from chromadb import PersistentClient
=======


>>>>>>> Stashed changes
import os, pathlib
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from numpy.linalg import norm
from PIL import Image

load_dotenv()

BATCH_SIZE = 32
CHROMA_BATCH_SIZE = 512
MODEL_PATH = './models/dungeon_model.keras'
DATASET_PATH = os.getenv("DATASET_PATH")
CHROMA_COLLECTION_NAME = 'sprite_embeddings'
CHROMA_DB_PATH = './chroma_db'

model, embedding_model = load_model(MODEL_PATH)
image_paths = load_image_paths(DATASET_PATH)
print(f"Found {len(image_paths)} images.")

def normalize_vec(vec):
    return vec / norm(vec)

data_dir = pathlib.Path(DATASET_PATH)

def get_label_from_path(file_path):
    path_object = pathlib.Path(file_path)
    return str(path_object.parent.relative_to(data_dir))

def load_and_process_image(path):
    def _process(p):
        img = Image.open(p.numpy().decode("utf-8")).convert("RGB")
        img = img.resize((32, 32))
        return np.array(img, dtype=np.float32)

    image = tf.py_function(_process, [path], tf.float32)
    image.set_shape([32, 32, 3])
    return image

if __name__ == "__main__":
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_process_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    embeddings = embedding_model.predict(dataset)
    print("Embedding shape:", embeddings.shape)

    # Reset collection to ensure metadata update (wipes out cosine collection cleanly)
    print("Resetting collection...")
    client = PersistentClient(path=CHROMA_DB_PATH)
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"Deleted collection '{CHROMA_COLLECTION_NAME}'")
    except ValueError:
        print(f"Collection '{CHROMA_COLLECTION_NAME}' did not exist.")

    collection = get_chroma_collection(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)

    labels = [get_label_from_path(p) for p in image_paths]

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
