import os
import numpy as np
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import tensorflow as tf
# import requests
# import zipfile
# import shutil
# from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import pathlib
import datetime
# from glob import glob

# def download_flatten_clean(url, workspace_dir="./"):
#     """
#     Downloads the .zip file dataset from the given url and then prepares it for processing.
#     It takes all the recursive sub-directories and flattens them out into the top directory only.
#     This makes it easier for us to process later.

#     Each sub-directory is separated by a '|', since '-' is used by some of the sub-directory names. 
#     """

#     workspace = Path(workspace_dir)
#     workspace.mkdir(parents=True, exist_ok=True)
#     zip_filename = url.split("/")[-1] or "downloaded_file.zip"
#     zip_path = workspace / zip_filename
#     raw_extract_path = workspace / "extract"
#     flattened_path = workspace / "training_data"

#     # Download Check
#     if zip_path.exists():
#         print(f"Zip file '{zip_filename}' already exists. Skipping download.")
#     else:
#         print(f"Downloading {url}...")
#         try:
#             response = requests.get(url, stream=True)
#             response.raise_for_status()
#             with open(zip_path, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
#         except Exception as e:
#             print(f"Download failed: {e}")
#             return

#     # Extraction Check
#     if not raw_extract_path.exists():
#         print("Extracting zip file...")
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(raw_extract_path)

#     # Create Flattened Structure
#     if not flattened_path.exists():
#         flattened_path.mkdir(exist_ok=True)
#         print("Flattening folders and renaming...")
        
#         for path in raw_extract_path.rglob('*'):
#             if path.is_dir():
#                 relative_parts = path.relative_to(raw_extract_path).parts
#                 if len(relative_parts) <= 1:
#                     continue
                
#                 # Creating the new flattened folder, subcategories separated by |
#                 new_folder_name = "|".join(relative_parts[1:])
#                 target_dir = flattened_path / new_folder_name
#                 shutil.copytree(path, target_dir, dirs_exist_ok=True)

#         # Cleanup Subdirectories within the Flattened Folder
#         print("Cleaning up nested subdirectories in the flattened results...")
#         for folder in flattened_path.iterdir():
#             if folder.is_dir():
#                 # Look for any sub-folders inside 
#                 for subpath in list(folder.rglob('*')):
#                     if subpath.is_dir():
#                         shutil.rmtree(subpath)
        
#         print("Processing complete. Ready for training.")
#     else:
#         print(f"Flattened folder already exists.")
    
#     return flattened_path

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
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        layers.RandomBrightness(0.05),
        layers.RandomZoom(0.05),
        layers.RandomRotation(0.02)
    ]

    for layer in data_augmentation_layers:
        images = layer(images)
    return images

def build_model(num_classes):
    """
    """

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

    x = layers.Conv2D(512, (1, 1), padding="same", activation="relu")(x)

    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])

    x = layers.Dense(
        1024, 
        kernel_initializer="he_normal", 
        kernel_regularizer=keras.regularizers.l2(0.001),
        kernel_constraint=keras.constraints.MaxNorm(3)
    )(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.5)(x) 

    x = layers.Dense(
        512, 
        kernel_initializer="he_normal", 
        kernel_regularizer=keras.regularizers.l2(0.001),
        kernel_constraint=keras.constraints.MaxNorm(3)
    )(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(
        inputs = inputs,
        outputs = outputs,
        name = "dungeon_archivist_classifier"
    )

    return model


if __name__ == "__main__":
    load_dotenv()
    URL = os.getenv("DATASET_LINK")
    DATASET_PATH = os.getenv("DATASET_PATH") #download_flatten_clean(URL, "./")
    BATCH_SIZE = 32
    LOG_DIR = f"logs/test/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    EPOCHS = 100

    clean_images(DATASET_PATH)

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=(32,32),
        batch_size=BATCH_SIZE,
        color_mode="rgba",
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    model = build_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.2, 
            patience=3, 
            min_lr=1e-7
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15, # Stop if no improvement for 15 epochs
            restore_best_weights=True
        )
    ]

    y_train = np.concatenate([np.argmax(y, axis=1) for x, y in train_ds], axis=0)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(weights))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    print(f"Done training.")


    



    







    
    


