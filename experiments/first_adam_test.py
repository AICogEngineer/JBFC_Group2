import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import datetime
import os
import pathlib
from dotenv import load_dotenv
import numpy as np

# env_path = os.path.join(os.path.dirname(__file__), '.env')
# load_dotenv(dotenv_path=env_path)

if __name__ == "__main__":
    load_dotenv()
    DATASET_PATH = pathlib.Path(os.getenv("DATASET_PATH"))
    IMG_SIZE = (32,32)
    BATCH_SIZE = 32

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

    # Pre-process and normalize images and attach integer label from path label
    def load_image(filepath, label):
        img = tf.io.read_file(filepath)
        img = tf.io.decode_png(img, channels=3) 
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        label_index = label_lookup(label)
        return img, label_index

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(image_paths), seed=42)

    dataset_batches = len(dataset)
    train_size = int(0.8 * dataset_batches)
    train_ds = dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_adam_model(learning_rate):
    adam_model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu", name="feature_embedding"),
        layers.Dense(num_classes, activation="softmax")
    ])

    adam_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return adam_model

adam_model =build_adam_model(learning_rate=0.0001)

log_dir= "logs/Adams/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

early_stopping =EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

adam_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[tensorboard_callback, early_stopping]
)