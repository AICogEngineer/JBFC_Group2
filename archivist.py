from project_utils import archivist_utils
import os
from pathlib import Path
import shutil
from collections import Counter

# Archivist Variables
MODEL_PATH = "output/models/dungeon_model_ab.keras"
TRAINED_DATASET_PATH = "datasets/test_ab"
SORT_DATASET_PATH = "datasets/dataset_c-main"
CHROMA_COLLECTION_NAME = "sprite_embeddings"
CHROMA_DB_PATH = "./chroma_db"
DISTANCE_THRESHOLD = 0.8
OUTPUT_DIR = "output/model_sorted_dataset"


if __name__ == "__main__":
    archivist_utils.load_sprite_embeddings_into_chromadb(MODEL_PATH, TRAINED_DATASET_PATH)

    model, embedding_model = archivist_utils.load_model(MODEL_PATH)
    image_paths = archivist_utils.load_image_paths(SORT_DATASET_PATH)
    collection = archivist_utils.get_chroma_collection(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)

    def get_label_from_path(file_path, dataset_root):
        path_obj = Path(file_path)
        relative_path = path_obj.parent.relative_to(dataset_root)
        return str(relative_path)

    for query_sprite in image_paths:
        query_vector = archivist_utils.get_embedding_from_file(query_sprite, embedding_model)
        query_vector = archivist_utils.normalize_vec(query_vector)
        results = collection.query(query_embeddings=[query_vector], n_results=5)

        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        if (sum(distances) / len(distances)) > DISTANCE_THRESHOLD:
            predicted_label = "Unlabeled"
        else:
            retrieved_labels = [meta["label"] for meta in metadatas]
            vote_counts = Counter(retrieved_labels)
            predicted_label = vote_counts.most_common(1)[0][0]
        
        label_parts = predicted_label.split("|")
        dest_folder = Path(OUTPUT_DIR, *label_parts)
        dest_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(query_sprite, dest_folder)
    
    print("Completed sorting.")


    



    