import os
import numpy as np
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalFocalCrossentropy
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import requests
import zipfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import pathlib
import datetime
from glob import glob

def download_flatten_clean(url, workspace_dir="./"):
    """
    Downloads the .zip file dataset from the given url and then prepares it for processing.
    It takes all the recursive sub-directories and flattens them out into the top directory only.
    This makes it easier for us to process later.

    Each sub-directory is separated by a '|', since '-' is used by some of the sub-directory names. 
    """

    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    zip_filename = url.split("/")[-1] or "downloaded_file.zip"
    zip_path = workspace / zip_filename
    raw_extract_path = workspace / "extract"
    flattened_path = workspace / "training_data"

    # Download Check
    if zip_path.exists():
        print(f"Zip file '{zip_filename}' already exists. Skipping download.")
    else:
        print(f"Downloading {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Download failed: {e}")
            return

    # Extraction Check
    if not raw_extract_path.exists():
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_extract_path)

    # Create Flattened Structure
    if not flattened_path.exists():
        flattened_path.mkdir(exist_ok=True)
        print("Flattening folders and renaming...")
        
        for path in raw_extract_path.rglob('*'):
            if path.is_dir():
                relative_parts = path.relative_to(raw_extract_path).parts
                if len(relative_parts) <= 1:
                    continue
                
                # Creating the new flattened folder, subcategories separated by |
                new_folder_name = "|".join(relative_parts[1:])
                target_dir = flattened_path / new_folder_name
                shutil.copytree(path, target_dir, dirs_exist_ok=True)

        # Cleanup Subdirectories within the Flattened Folder
        print("Cleaning up nested subdirectories in the flattened results...")
        for folder in flattened_path.iterdir():
            if folder.is_dir():
                # Look for any sub-folders inside 
                for subpath in list(folder.rglob('*')):
                    if subpath.is_dir():
                        shutil.rmtree(subpath)
        
        print("Processing complete. Ready for training.")
    else:
        print(f"Flattened folder already exists.")
    
    return flattened_path

def clean_images(DATASET_PATH):
    """
    Removes the ICC profile and all other not relevant metadata from the images.
    """
    DATASET_PATH = pathlib.Path(DATASET_PATH)

    for img_path in DATASET_PATH.rglob("*.png"):
        img = Image.open(img_path)
        img = img.convert("RGBA")
        img.save(img_path, icc_profile=None)

def process_images_and_create_encodings(DATASET_PATH):
    """
    Returns X(images), Y(One Hot Encodings of the Labels), Y_indices(label indices for class weight calculations), and encoders(for layer size information).
    """

    images = []
    labels_topCategory = []
    labels_subCategory = []
    labels_sub_subCategory = []

    # Create array with folder paths
    folders = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    print(f"Found {len(folders)} folders. Parsing hierarchy...")

    """
    We need to split and save each image's category and sub-categories to be put into the CNN model.
    To help training, we normalize the image RGBA data (0-255 to 0-1).
    """

    for folder in folders:
        folder_path = os.path.join(DATASET_PATH, folder)
        category_parts = folder.split("|")
        topCategory = category_parts[0]
        subCategory = category_parts[1] if len(category_parts) > 1 else "Generic"
        sub_subCategory = category_parts[2] if len(category_parts) > 2 else "Generic"

        files = glob(os.path.join(folder_path, "*.[pP][nN][gG]"))
        for file_path in files:
            image_file = tf.io.read_file(file_path)
            image_file = tf.io.decode_png(image_file, channels=4)
            image_file = tf.cast(image_file, tf.float32) / 255.0
            
            images.append(image_file)
            labels_topCategory.append(topCategory)
            labels_subCategory.append(subCategory)
            labels_sub_subCategory.append(sub_subCategory)

    if not images:
        raise ValueError("No images found. Check path.")

    print(f"Loaded {len(images)} images.")

    """
    In this training, we use CategoricalFocalCrossentropy to deal with the imbalanced dataset given.
    CategoricalFocalCrossentropy expects the labels to be in One Hot Encoding.
    While One Hot Encoding is not ideal, the amount of categories we are dealing with are not massively large.

    Before training, we need to turn the string category labels into integers which we can then turn into 
    One Hot Encoding. The easiest way to do this is to use SciKit LabelEncoder's fit_transform().
    fit_transform() will learn all the unique categories and convert them into normalized integers.
    We can then turn those integers into One Hot Encodings.
    """

    labelEncoder_topCategory = LabelEncoder()
    labelEncoder_subCategory = LabelEncoder()
    labelEncoder_sub_subCategory = LabelEncoder()

    y_topCategory_idx = labelEncoder_topCategory.fit_transform(labels_topCategory)
    y_subCategory_idx = labelEncoder_subCategory.fit_transform(labels_subCategory)
    y_sub_subCategory_idx = labelEncoder_sub_subCategory.fit_transform(labels_sub_subCategory)

    y_topCategory = keras.utils.to_categorical(y_topCategory_idx, num_classes=(len(labelEncoder_topCategory.classes_)))
    y_subCategory = keras.utils.to_categorical(y_subCategory_idx, num_classes=(len(labelEncoder_subCategory.classes_)))
    y_sub_subCategory = keras.utils.to_categorical(y_sub_subCategory_idx, num_classes=(len(labelEncoder_sub_subCategory.classes_)))
    X = np.array(images)

    encoders = {
        "topCategory": labelEncoder_topCategory,
        "subCategory": labelEncoder_subCategory,
        "sub_subCategory": labelEncoder_sub_subCategory
    }

    Y = {
        "output_topCategory": y_topCategory,
        "output_subCategory": y_subCategory,
        "output_sub_subCategory": y_sub_subCategory
    }

    Y_indices = {
        "output_topCategory": y_topCategory_idx,
        "output_subCategory": y_subCategory_idx,
        "output_sub_subCategory": y_sub_subCategory_idx
    }

    return X, Y, Y_indices, encoders

def custom_pixel_data_augmentation_layer():
    """
    Returns the sequential data augmentation layer made for pixel art.

    Using normal data augmentation methods will use Anti-Aliasing by default when doing transformations.
    To deal with this, we need to define our own using Nearest Neighbor instead.
    """

    return keras.Sequential([
        layers.Input(shape=(32, 32, 4)),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.Lambda(lambda x: tf.image.rot90(x, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        layers.Resizing(32, 32, interpolation="nearest")
    ], name="pixel_data_augmentation")

def build_model(NUM_TOPCATEGORIES, NUM_SUBCATEGORIES, NUM_SUB_SUBCATEGORIES):
    """
    Builds and returns the multi-input/output CNN model.

    To help train our model to recognize each image's category and their respective sub-catgories (up to 3 in depth),
    we have to train on each category separately in their own branch. Each branch will receive the same base construction. However, their outputs will be different, since we put more weight onto the higher categories.

    To deal with overfitting and our highly imbalanced dataset, we use LeakyReLU, SpatialDropout2D and GlobalAveragePooling2D. Since our features are so minute due to the nature of pixel art, there is a chance we may have some dead neurons, so we use LeakyReLU. SpatialDropout will help with overfitting by dropping out some neurons randomly to provide redundancy. GlobalAveragePooling2D will keep more features to prevent overfitting as well.
    """

    inputs = keras.Input(shape=(32, 32, 4))
    aug = custom_pixel_data_augmentation_layer()(inputs)

    # Shared Base
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(aug)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.SpatialDropout2D(0.1)(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.SpatialDropout2D(0.3)(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    shared_features = layers.GlobalAveragePooling2D(name="shared_features")(x)

    # Top Category branch
    topCategory_branch = layers.Dense(64, activation="relu")(shared_features)
    output_topCategory = layers.Dense(NUM_TOPCATEGORIES, activation="softmax", name="output_topCategory")(topCategory_branch)

    # Sub Category branch
    subCategory_input = layers.Concatenate()([shared_features, topCategory_branch])
    subCategory_branch = layers.Dense(128, activation="relu")(subCategory_input)
    subCategory_branch = layers.Dropout(0.4)(subCategory_branch)
    output_subCategory = layers.Dense(NUM_SUBCATEGORIES, activation="softmax", name="output_subCategory")(subCategory_branch)

    # Sub-sub Category branch
    sub_subCategory_input = layers.Concatenate()([shared_features, subCategory_branch])
    sub_subCategory_branch = layers.Dense(128, activation="relu")(sub_subCategory_input)
    sub_subCategory_branch = layers.Dropout(0.5)(sub_subCategory_branch)
    output_sub_subCategory = layers.Dense(NUM_SUB_SUBCATEGORIES, activation="softmax", name="output_sub_subCategory")(sub_subCategory_branch)

    model = keras.Model(
        inputs = inputs,
        outputs = [output_topCategory, output_subCategory, output_sub_subCategory],
        name = "dungeon_archivist_classifier"
    )

    return model


if __name__ == "__main__":
    load_dotenv()
    URL = os.getenv("DATASET_LINK")
    DATASET_PATH = download_flatten_clean(URL, "./")
    BATCH_SIZE = 32
    LOG_DIR = f"logs/test/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    EPOCHS = 50

    clean_images(DATASET_PATH)
    X_train, Y_train_onehot, Y_train_indices, encoders = process_images_and_create_encodings(DATASET_PATH)

    Y_dict = {
    "output_topCategory": Y_train_onehot["output_topCategory"],
    "output_subCategory": Y_train_onehot["output_subCategory"],
    "output_sub_subCategory": Y_train_onehot["output_sub_subCategory"]
}

    NUM_TOPCATEGORY = len(encoders["topCategory"].classes_)
    NUM_SUBCATEGORY = len(encoders["subCategory"].classes_)
    NUM_SUB_SUBCATEGORY = len(encoders["sub_subCategory"].classes_)
    print(f"Detected {NUM_TOPCATEGORY} top categories, {NUM_SUBCATEGORY} subcategories, and {NUM_SUB_SUBCATEGORY} sub-subcategories.")

    X_train, X_val, idx_train, idx_val = train_test_split(
        X_train, 
        np.arange(len(X_train)), # We split indices to slice the Y dictionary later
        test_size=0.2, 
        random_state=42,
        shuffle=True # This ensures classes are mixed!
    )

    # Helper to slice the dictionary based on indices
    def slice_dict(y_dict, indices):
        return {k: v[indices] for k, v in y_dict.items()}

    Y_train = slice_dict(Y_dict, idx_train)
    Y_val = slice_dict(Y_dict, idx_val)

    focal_loss = CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0, from_logits=False)

    model = build_model(NUM_TOPCATEGORY, NUM_SUBCATEGORY, NUM_SUB_SUBCATEGORY)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "output_topCategory": "categorical_crossentropy",
            "output_subCategory": focal_loss,
            "output_sub_subCategory": focal_loss
        },
        loss_weights={
            "output_topCategory": 1.0,   # Base priority
            "output_subCategory": 1.5,   # Higher priority
            "output_sub_subCategory": 2.5 # Highest priority (fix the imbalance here)
        },
        metrics={
            "output_topCategory": ["accuracy"], # Simple is fine
            "output_subCategory": ["accuracy"], # Check if it's close
            "output_sub_subCategory": ["accuracy"] # Check for imbalance cheating
        }
    )

    callbacks = [
        # Stop if the hardest task (subsub) stops improving
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6
        )
    ]

    model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    print(f"Done training.")


    



    







    
    


