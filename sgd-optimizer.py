import os
# Suppress TensorFlow INFO and WARNING logs -->> those annoying PNG iCCP warnings >:(
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import pathlib
import random

# ==========================================
# Configuration
# ==========================================
IMG_HEIGHT = 32
IMG_WIDTH = 32
BATCH_SIZE = 32
EMBEDDING_DIM = 128
DATA_DIR = './data/Dungeon Crawl Stone Soup Full_v2/'

# ==========================================
# Step 1: Data Loading
# ==========================================
# Use a manual approach here because 'image_dataset_from_directory'
# doesn't see the deep subfolders (like 'monster/abyss') by default.
print(f"Loading data from: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory not found at {DATA_DIR}")
    exit()

# 1.1 Find all image files
data_dir = pathlib.Path(DATA_DIR)
# 'rglob' stands for Recursive Glob (searches all subfolders recursively)
image_paths = list(data_dir.rglob('*.png'))
image_paths = [str(path) for path in image_paths] # Convert Path objects to strings

# 1.2 Shuffle the data
# Shuffle so the model doesn't learn order (e.g. all monsters, then all items)
random.seed(13)
random.shuffle(image_paths)

print(f"Found {len(image_paths)} images.")

# 1.3 Get Labels aka the folder name
# The label is the relative path to the folder the image is in.
# Example: /.../monster/abyss/crab.png -> Label: 'monster/abyss'
def get_label_from_path(file_path):
    path_object = pathlib.Path(file_path)
    # Get path relative to the data root
    relative_path = path_object.parent.relative_to(data_dir)
    return str(relative_path)

# Create a list of labels matching our list of images
labels = [get_label_from_path(path) for path in image_paths]

# Find all unique class names (the sorted list of labels)
class_names = sorted(list(set(labels)))
print(f"Found {len(class_names)} classes: {class_names}")

# 1.4 Convert name labels to numbers
label_to_index = {name: i for i, name in enumerate(class_names)}
all_labels = [label_to_index[lbl] for lbl in labels]

# 1.5 Split Data (70% Train, 30% Validation)
val_size = int(len(image_paths) * 0.3)

train_paths = image_paths[val_size:]   # Last 70%
train_labels = all_labels[val_size:]

val_paths = image_paths[:val_size]     # First 30%
val_labels = all_labels[:val_size]

print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")

# 1.6 Helper function to load and fix images
def load_and_process_image(path, label):
    # Read the file from disk
    image = tf.io.read_file(path)
    # Decode PNG format
    image = tf.image.decode_png(image, channels=3)
    # Resize just in case (hidden boss thing might be random dimensions)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    # Normalize: Convert pixel values from 0-255 to 0-1
    image = image / 255.0
    return image, label

# 1.7 Create TensorFlow Datasets
# inputs, targets
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(load_and_process_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_and_process_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

print("Data is ready :)")


# ==========================================
# Step 2: Model Architecture
# ==========================================
num_classes = len(class_names)

def create_model():
    model = keras.Sequential([
        # Input Layer: 32 filters, looks for small patterns (edges, corners)
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)), # Shrinks the image size
        
        # Second Layer: 64 filters, looks for shapes and patterns
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Third Layer: 128 filters, looks for complex patterns
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten: Turn the 2D grid into 1D
        layers.Flatten(),
        
        # Hidden Layer: 128 neurons
        layers.Dense(128, activation='relu'),

        # Embedding layer for similiarity search
        layers.Dense(EMBEDDING_DIM, activation=None, name="embedding"),
        
        # Output Layer: softmax for classification
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ==========================================
# Step 3: Training Loop
# ==========================================
learning_rates = [0.01, 0.001, 0.05]

print("\nStarting Training Experiment...")
for lr in learning_rates:
    print(f"\n--- Training with Learning Rate: {lr} ---")
    
    # Initialize Model
    sgd_model = create_model()

    # Configure Optimizer (SGD)
    sgd_model.compile(
        optimizer = keras.optimizers.SGD(learning_rate=lr),
        loss = "sparse_categorical_crossentropy", # Standard for integer labels
        metrics = ["accuracy"]
    )

    # Setup Logs and Callbacks
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOGS_DIR = f"logs/training/sgd_lr_{lr}_{current_time}"
    
    # TensorBoard: Visualizes the training graphs
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR, histogram_freq=1)
    
    # Early Stopping: Stops training if it's not getting better
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Run Training
    history = sgd_model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = 50,
        callbacks = [tensorboard_callback, early_stopping]
    )
    
    print(f"Completed LR {lr}. Logs saved to {LOGS_DIR}")
    
    # Save the model for ChromaDB integration
    if lr == 0.01: # Save the one with 0.01 LR as our 'production' model
        model_save_path = "./models/dungeon_model.keras"
        sgd_model.save(model_save_path)
        print(f"Model saved to {model_save_path}")