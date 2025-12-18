import os
import numpy as np
import keras
from keras import layers
import tensorflow as tf
import datetime
import pathlib

# dataset_path = pathlib.Path("training_data/Dungeon Crawl Stone Soup Full")
# img_size = (32,32)
# batch_size = 32

# image_paths = list(dataset_path.rglob("*.png"))

# labels = []
# for path in image_paths:
#     p = pathlib.Path(path)
#     relative_path = p.relative_to(dataset_path)
#     folder_structure = relative_path.parent
#     label = "_".join(folder_structure.parts)
#     labels.append(label)

# image_paths = [str(p) for p in image_paths]

# print(f"Found {len(image_paths)} images.")
# if len(image_paths) > 0:
#     print(f"Sample Label: {labels[0]}")

# label_lookup = layers.StringLookup(output_mode="int")
# label_lookup.adapt(labels)

# class_names = label_lookup.get_vocabulary()
# num_classes = label_lookup.vocabulary_size()
# print(f"Unique classes found: {num_classes}")

# def load_image(filepath, label):
#     """Reads image file and converts string label to integer index."""
#     img = tf.io.read_file(filepath)
#     # Decode PNG (set channels=3 to force RGB even if image is Grayscale)
#     img = tf.io.decode_png(img, channels=3) 
#     img = tf.image.resize(img, img_size)
#     img = tf.cast(img, tf.float32) / 255.0 # Normalize 0-1
    
#     # Convert string label to integer index
#     label_index = label_lookup(label)
#     return img, label_index

# dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
# dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
# dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(batch_size)
# dataset = dataset.prefetch(tf.data.AUTOTUNE)

# dataset_batches = len(dataset)
# train_size = int(0.8 * dataset_batches)
# train_ds = dataset.take(train_size)
# val_ds = dataset.skip(train_size)

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "training_data/flattenset",
    validation_split=0.2,
    subset="both",
    seed=42,
    image_size=(32,32),
    batch_size=32,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Detected categories: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

sgd_momentum_model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32,(3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

sgd_momentum_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

log_dir = "logs/sgd_momentum/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
    )

history = sgd_momentum_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[tensorboard_callback]
)

sgd_momentum_model.summary()

sgd_momentum_model.save("test.keras")
print("Finished")