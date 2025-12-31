import os
import numpy as np
import keras
from keras import layers
from keras.callbacks import EarlyStopping
import tensorflow as tf
import datetime
import pathlib
from dotenv import load_dotenv
from PIL import Image

if __name__ == "__main__":
    load_dotenv()
    DATASET_PATH = pathlib.Path(os.getenv("DATASET_PATH"))
    IMG_SIZE = (32,32)
    BATCH_SIZE = 32

    for img_path in DATASET_PATH.rglob("*.png"):
        img = Image.open(img_path)
        img = img.convert("RGBA")
        img.save(img_path, icc_profile=None)

    # Load in all paths inside the training set as their own categories
    image_paths = list(DATASET_PATH.rglob("*.png"))
    labels = []
    for path in image_paths:
        p = pathlib.Path(path)
        relative_path = p.relative_to(DATASET_PATH)
        folder_structure = relative_path.parent
        label = "_".join(folder_structure.parts)
        labels.append(label)
    image_paths = [str(p) for p in image_paths]
    print(f"Found {len(image_paths)} images.") 

    label_lookup = layers.StringLookup(output_mode="int")
    label_lookup.adapt(labels)
    class_names = label_lookup.get_vocabulary()
    num_classes = label_lookup.vocabulary_size()
    print(f"Unique classes found: {num_classes}")

    import csv
    from collections import Counter

    # --- START OF CSV LOGGING WITH SUMMARY ---
    csv_file_path = "dataset_manifest.csv"
    
    # 1. Pre-calculate the IDs for all labels
    # We pass the list of labels through the lookup layer and convert to numpy integers
    numeric_ids = label_lookup(labels).numpy()
    
    # 2. Calculate counts for the summary table
    label_counts = Counter(labels)

    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # --- SECTION 1: FULL DATA MANIFEST ---
        writer.writerow(["File Path", "Category Label", "Class ID"])
        for path, label, class_id in zip(image_paths, labels, numeric_ids):
            writer.writerow([path, label, int(class_id)])
        
        # --- SECTION 2: SUMMARY TABLE ---
        writer.writerow([])  # Add an empty row as a visual separator
        writer.writerow(["--- SUMMARY TABLE ---"])
        writer.writerow(["Class ID", "Category Label", "Total Image Count"])
        
        # We iterate through class_names (the vocabulary) to ensure the summary 
        # is sorted by Class ID order.
        for i, name in enumerate(class_names):
            # label_counts uses the string name as the key
            count = label_counts.get(name, 0)
            writer.writerow([i, name, count])

    print(f"Dataset manifest and summary successfully saved to {csv_file_path}")
    # --- END OF CSV LOGGING ---

    # Pre-process and normalize images and attach integer label from path label
    def load_image(filepath, label):
        img = tf.io.read_file(filepath)
        img = tf.io.decode_png(img, channels=4) 
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        label_index = label_lookup(label)
        return img, label_index

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(image_paths), seed=42, reshuffle_each_iteration=False)

    dataset_batches = len(dataset)
    train_size = int(0.8 * dataset_batches)
    train_ds = dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def build_sgd_momentum_model(learning_rate, momentum):
        sgd_momentum_model = keras.Sequential([
            keras.Input(shape=(32, 32, 4)),

            layers.Conv2D(32,(3,3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),

            layers.Conv2D(64,(3,3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),

            layers.Conv2D(128,(3,3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),

            layers.Flatten(),
            layers.Dense(128, activation="relu", name="feature_embedding"),
            layers.Dense(num_classes, activation="softmax")
        ])

        sgd_momentum_model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return sgd_momentum_model
    
    learning_rates = [0.001, 0.01, 0.1]
    momentum_values = [0.1, 0.5, 0.9]
    epochs = [50]

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
    )

    # for lr in learning_rates:
    #     for momentum in momentum_values:
    #         for epoch in epochs:
    #             sgd_momentum_model = build_sgd_momentum_model(lr, momentum)
    #             log_dir = f"logs/sgd_momentum_new_dataset_test/{lr}/{momentum}/{epoch}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #             tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #                 log_dir=log_dir,
    #                 histogram_freq=1
    #                 )
    #             sgd_momentum_model.fit(
    #                 train_ds,
    #                 validation_data=val_ds,
    #                 epochs=epoch,
    #                 callbacks=[tensorboard_callback, early_stopping]
    #             )

    print("Finished")