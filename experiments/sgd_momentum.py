import os
import numpy as np
import keras
from keras import layers
import tensorflow as tf
import datetime
import pathlib
from dotenv import load_dotenv

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

    def build_sgd_momentum_model(learning_rate, momentum):
        sgd_momentum_model = keras.Sequential([
            keras.Input(shape=(32, 32, 3)),

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
    momentum_values = [0.1, 0.5, 0.9, 0.99]
    epochs = [15, 30, 50]

    for lr in learning_rates:
        for momentum in momentum_values:
            for epoch in epochs:
                sgd_momentum_model = build_sgd_momentum_model(lr, momentum)
                log_dir = f"logs/sgd_momentum/{lr}/{momentum}/{epoch}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                tensorboard_callback = tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1
                    )
                sgd_momentum_model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=epoch,
                    callbacks=[tensorboard_callback]
                )

    print("Finished")