import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import shutil
import datetime

# set constants/variables
IMG_HEIGHT = 32
IMG_WIDTH = 32
BATCH_SIZE = 32
DATA_DIR = './data/Dungeon Crawl Stone Soup Full/'
LOG_DIR_SAMPLE = "logs/data_inspection/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# Data Loading
print(f"Loading data from: {DATA_DIR}")
print(f"TensorBoard sample logs will be saved to: {LOG_DIR_SAMPLE}")

if not os.path.exists(DATA_DIR):
    print(f"Data directory not found at {DATA_DIR}")
else:
    # Training split (70%)
    train_ds = tf.keras.utils.image_dataset_from_directory( # standard way to load data from a directory and label it by subdirectories
        DATA_DIR,
        validation_split=0.3,
        subset="training",
        seed=13,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # Validation split (30%)
    val_ds = tf.keras.utils.image_dataset_from_directory( # standard way to load data from a directory and label it by subdirectories
        DATA_DIR,
        validation_split=0.3,
        subset="validation",
        seed=13,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # This is how we find out the class names in our directory, set it to a variable for later use
    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")

    # Tensorboard stuff
    print("writing sample images to TensorBoard...")
    
    # Create a file writer for the log directory
    file_writer = tf.summary.create_file_writer(LOG_DIR_SAMPLE)

    # Get a batch of images
    for images, labels in train_ds.take(1):
        # takes a single batch of images and labels to view in TensorBoard to ensure the data is loaded correctly
        
        with file_writer.as_default():
            tf.summary.image("Training Data Examples", images, step=0, max_outputs=25)
            
    print(f"Run: tensorboard --logdir={LOG_DIR_SAMPLE}") # I always forget this syntax

    # 3. Preprocessing (Normalization)
    normalization_layer = layers.Rescaling(1./255) # scales the pixel values to be between 0 and 1 instead of 0-255
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # apply the normalization to the training dataset
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)) # apply the normalization to the validation dataset


    print("Data is ready :)")


# making the model

num_classes = len(class_names)

sgd_model = keras.Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# TODO: Compile model
sgd_model.compile(
    optimizer = keras.optimizers.SGD(learning_rate=0.01),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

LOGS_DIR = "logs/training/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR, histogram_freq=1)

# TODO: Train model
history = sgd_model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 10,
    callbacks = [tensorboard_callback]
)