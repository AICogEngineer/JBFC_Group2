import os
import numpy as np
import keras
from keras import layers
import tensorflow as tf
import datetime


#Should we account for the alpha channel too?

dataset_path = "training_data/Dungeon Crawl Stone Soup Full"
img_size = (32,32)
batch_size = 32

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="both",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"# of Categories: {class_names}")

sgd_momentum_model = keras.Sequential([
    layers.Conv2D(32,(3,3), activation="relu", input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

sgd_momentum_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

log_dir = "logs/sgd_momentum/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
    )

sgd_momentum_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback]
)

sgd_momentum_model.save("test.keras")
print("Finished")