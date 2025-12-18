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
LOG_DIR = "logs/data_inspection/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# Data Loading
print(f"Loading data from: {DATA_DIR}")
print(f"TensorBoard logs will be saved to: {LOG_DIR}")

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
    file_writer = tf.summary.create_file_writer(LOG_DIR)

    # Get a batch of images
    for images, labels in train_ds.take(1):
        # takes a single batch of images and labels to view in TensorBoard to ensure the data is loaded correctly
        
        with file_writer.as_default():
            tf.summary.image("Training Data Examples", images, step=0, max_outputs=25)
            
    print(f"Run: tensorboard --logdir={LOG_DIR}") # I always forget this syntax

    # 3. Preprocessing (Normalization)
    normalization_layer = layers.Rescaling(1./255) # scales the pixel values to be between 0 and 1 instead of 0-255
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # apply the normalization to the training dataset
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)) # apply the normalization to the validation dataset


    print("Data is ready :)")


    # TODO: Make model
    # TODO: Create optimizer
    # TODO: Compile model
    # TODO: Train model

