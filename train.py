import os
import numpy as np
import keras
from sklearn.utils import class_weight
import tensorflow as tf
import datetime
from project_utils import train_utils

if __name__ == "__main__":

    # Training Variables to change
    DATASET_PATH = "datasets/test_ab_og"
    BATCH_SIZE = 32
    LOG_DIR = f"logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-ab-og-128"
    EPOCHS = 100
    MODEL_DIR = "output/models_present/"
    MODEL_NAME = "dungeon_model_ab_og_present.keras"

    train_utils.clean_images(DATASET_PATH)

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="both",
        seed=42,
        color_mode="rgba",
        image_size=(32,32),
        batch_size=BATCH_SIZE,
        labels="inferred", 
        label_mode="int"  
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    model = train_utils.build_model(num_classes)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.2, 
            patience=3, 
            min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10, 
            restore_best_weights=True
        )
    ]

    y_train = np.concatenate([y for x, y in train_ds], axis=0)
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

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, MODEL_NAME))

    print(f"The training has completed, and the model has been saved to {os.path.join(MODEL_DIR, MODEL_NAME)}.")