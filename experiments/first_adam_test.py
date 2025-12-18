import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import os

dataset = tf.keras.utils.image_dataset_from_directory(
    './Dungeon Crawl Stone Soup Full',
    labels='inferred',
    label_mode='int', 
    image_size=(32, 32),
    batch_size=32
    validation_split=0.3,
)

(x_train,y_train),(x_test,y_test)=dataset

x_train = x_train.reshape(-1, 3072).astype('float32') / 255.0
x_test = x_test.reshape(-1, 3072).astype('float32') / 255.0

x_train_sub = x_train[:10000]
y_train_sub = y_train[:10000]


def create_model():
    """Create identical model for fair comparison"""
    return keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(32,32,3)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax') # will change later when I know the output 
    ])

log_dir = "logs/Adams/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

model = create_model()
model.compile(optimizer="Adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
history = model.fit(
    x_train_sub, y_train_sub,
    epochs=10,
    verbose=0,
    callbacks=[tensorboard_callback]
)