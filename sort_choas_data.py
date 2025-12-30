from sklearn.cluster import KMeans
import tensorflow as tf
import os
import pathlib
import keras
from keras import layers
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import shutil
from PIL import Image
from sklearn.preprocessing import StandardScaler

load_dotenv()
CHAOS_PATH = pathlib.Path(os.getenv("CHAOS_PATH"))
BACKGROUND_PATH = pathlib.Path(os.getenv("BACKGROUND_PATH"))

image_paths = list(CHAOS_PATH.rglob("*.png"))
no_background= list(BACKGROUND_PATH.rglob("*.png"))

print("Found images:", len(image_paths))

# def remove_background(img, threshold=0.95):
#     # img: (H, W, 3), values in [0,1]
#     gray = tf.image.rgb_to_grayscale(img)
#     mask = tf.cast(gray < threshold, tf.float32)
#     return img * mask

# def no_background(path):
#     img = load_image_rgb(path)
#     img = remove_background(img)
#     return img

# def load_image_gray(path):
#     img = no_background(path)
#     img = tf.image.rgb_to_grayscale(img) 
#     return img

# def load_rb_image(path):
#     img = tf.io.read_file(str(path))
#     img = tf.io.decode_png(img, channels=4) # regular rgb has 3 with trasnparency its 4
#     img = tf.image.resize(img, (32, 32))
#     img = tf.cast(img, tf.float32) / 255.0
#     #img = remove_background(img)
#     return img

def load_image_rgb(path):
    img = tf.io.read_file(str(path))
    img = tf.io.decode_png(img, channels=4)
    img = tf.image.resize(img, (32, 32))
    img = tf.cast(img, tf.float32) / 255.0
    return img

def add_noise(img):
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.1)
    return tf.clip_by_value(img + noise, 0.0, 1.0)


images = tf.stack([load_image_rgb(p) for p in no_background])

dataset = tf.data.Dataset.from_tensor_slices(images)
#dataset = dataset.map(lambda x: (x, x))
dataset = dataset.map(lambda x: (add_noise(x), x))
dataset = dataset.shuffle(buffer_size=len(images), seed=42,reshuffle_each_iteration=False)

train_size = int(0.8 * len(images))

train_ds = dataset.take(train_size).batch(32)
test_ds = dataset.skip(train_size).batch(32)

encoding_dim = 32
input_img = keras.Input(shape=(32, 32, 4)) # chage last value either to 1 or 3 depending
                                            # 1=grayscale   3=rgb

x = layers.Conv2D(32, 3, activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D(2, padding="same")(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)

x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)

latent = layers.Flatten()(x)
latent = layers.Dense(16, name="latent",kernel_regularizer=keras.regularizers.l2(1e-4))(latent)

x = layers.Dense(4 * 4 * 128)(latent)
x = layers.Reshape((4, 4, 128))(x)

x = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)

output_img = layers.Conv2D(4, 3, activation="sigmoid", padding="same")(x) #4,3 for transparancy 3,3 for normal


autoencoder = keras.Model(input_img, output_img)
autoencoder.compile(optimizer="adam", loss="mse")

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    min_delta=1e-4,
    restore_best_weights=True
)

autoencoder.fit(train_ds,
                epochs=50, #changes from 50 to 10
                batch_size=256,
                shuffle=True,
                validation_data=test_ds,
                callbacks=[early_stopping])



encoder = keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("latent").output)


all_images_rgb = np.stack([load_image_rgb(p).numpy() for p in image_paths])
background_rgb = np.stack([load_image_rgb(p).numpy() for p in no_background]) 

encoded_images = encoder.predict(background_rgb)
encoded_images = StandardScaler().fit_transform(encoded_images)

n=208 #changes from n from 22 to 208

kmeans = KMeans(n_clusters=n, random_state=42, n_init=10) 
cluster_labels = kmeans.fit_predict(encoded_images)

clustered_images = {i: [] for i in range(n)}  



OUTPUT_DIR = pathlib.Path("clustered_output")
OUTPUT_DIR.mkdir(exist_ok=True)

for i in range(n):
    (OUTPUT_DIR / f"cluster_{i}").mkdir(exist_ok=True)

for idx, label in enumerate(cluster_labels):
    img = all_images_rgb[idx]   # or all_images[idx]
    img = (img * 255).astype(np.uint8)

    Image.fromarray(img).save(
        OUTPUT_DIR / f"cluster_{label}" / f"img_{idx}.png"
    )



# inertias = []
# k_values = range(1, 15)

# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(encoded_train)
#     inertias.append(kmeans.inertia_)

# plt.figure(figsize=(8, 5))
# plt.plot(k_values, inertias, marker='o')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Inertia")
# plt.title("K-Means Elbow Method on Autoencoder Encodings")
# plt.show()


