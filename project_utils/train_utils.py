import os
import numpy as np
import keras
from keras import layers
from PIL import Image
import pathlib

def clean_images(DATASET_PATH):
    """
    Removes the ICC profile and all other not relevant metadata from the images.
    """
    DATASET_PATH = pathlib.Path(DATASET_PATH)

    for img_path in DATASET_PATH.rglob("*.png"):
        img = Image.open(img_path)
        img = img.convert("RGBA")
        img.save(img_path, icc_profile=None)

def data_augment(images):
    """
    Returns the sequential data augmentation images made for pixel art.
    """

    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomContrast(0.05),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomBrightness(0.05),
        layers.RandomZoom(0.05),
    ]

    for layer in data_augmentation_layers:
        images = layer(images)
    return images

def build_model(num_classes):

    inputs = keras.Input(shape=(32, 32, 4))
    x = data_augment(inputs)
    x = layers.Rescaling(1./255)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2,2))(x) 
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.5)(x) 

    embeddings = layers.Dense(128, activation=None, name="embeddings")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(embeddings)
    
    model = keras.Model(
        inputs = inputs,
        outputs = outputs,
        name = "dungeon_archivist_classifier"
    )

    return model

