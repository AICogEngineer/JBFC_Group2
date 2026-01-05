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
# grabs paths from .env for original chaos file & file with background turned transparent
CHAOS_PATH = pathlib.Path(os.getenv("CHAOS_PATH"))
BACKGROUND_PATH = pathlib.Path(os.getenv("BACKGROUND_PATH"))

# recurively grabs all of the png file from both folders
image_paths = list(CHAOS_PATH.rglob("*.png"))
no_background= list(BACKGROUND_PATH.rglob("*.png"))

print("Found images:", len(image_paths))

#loads the image
def load_image_rgb(path):
    img = tf.io.read_file(str(path))
    img = tf.io.decode_png(img, channels=4)
    img = tf.image.resize(img, (32, 32))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def add_noise(img):
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.1)
    return tf.clip_by_value(img + noise, 0.0, 1.0)

# loads images into variable
images = tf.stack([load_image_rgb(p) for p in no_background])

#create tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices(images)
#dataset = dataset.map(lambda x: (x, x))
dataset = dataset.map(lambda x: (add_noise(x), x)) # this was one of my initial way to discourage the backgroun fetaures
dataset = dataset.shuffle(buffer_size=len(images), seed=42,reshuffle_each_iteration=False)

#create the 80/20 split to train to encode the data
train_size = int(0.8 * len(images))
train_ds = dataset.take(train_size).batch(32)
test_ds = dataset.skip(train_size).batch(32)

encoding_dim = 32
input_img = keras.Input(shape=(32, 32, 4)) # chage last value either to 1 or 3 depending
                                            # 1=grayscale   3=rgb


# extract features 
#low
x = layers.Conv2D(32, 3, activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D(2, padding="same")(x)
#medium
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)
#high
x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)

latent = layers.Flatten()(x)
latent = layers.Dense(16, name="latent",kernel_regularizer=keras.regularizers.l2(1e-4))(latent)

#Decode
x = layers.Dense(4 * 4 * 128)(latent)
x = layers.Reshape((4, 4, 128))(x)
#reshape to original size
x = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)

# final output (how it would look decoded)
output_img = layers.Conv2D(4, 3, activation="sigmoid", padding="same")(x) #4,3 for transparancy 3,3 for normal

# compares the input and output and what the loss
autoencoder = keras.Model(input_img, output_img)
autoencoder.compile(optimizer="adam", loss="mse")

# incase the loss diverges it stops early
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    min_delta=1e-4,
    restore_best_weights=True
)
# the setting for traing the encoding
autoencoder.fit(train_ds,
                epochs=50, 
                batch_size=256,
                shuffle=True,
                validation_data=test_ds,
                callbacks=[early_stopping])


# calls the encoder for traing 
encoder = keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("latent").output)

# This second half is for using k-means to plot centriod which I would use to group images into files macking it easier
# to train

#look images into numpy
all_images_rgb = np.stack([load_image_rgb(p).numpy() for p in image_paths])
background_rgb = np.stack([load_image_rgb(p).numpy() for p in no_background]) 

encoded_images = encoder.predict(background_rgb)
encoded_images = StandardScaler().fit_transform(encoded_images)

n=208 #changes from n from 22 to 208

kmeans = KMeans(n_clusters=n, random_state=42, n_init=10) 
cluster_labels = kmeans.fit_predict(encoded_images) #this predict which group a image belongs too

clustered_images = {i: [] for i in range(n)}  # I might not need this as this was from older code

#create output directory namded clustereed output that what store the sorted images from chaos
OUTPUT_DIR = pathlib.Path("clustered_output")
OUTPUT_DIR.mkdir(exist_ok=True)

#create all the subfolders
for i in range(n):
    (OUTPUT_DIR / f"cluster_{i}").mkdir(exist_ok=True)

# start sorting each images based on it label to the subfolders
for idx, label in enumerate(cluster_labels):
    img = all_images_rgb[idx]   # or all_images[idx]
    img = (img * 255).astype(np.uint8)
    #save the images in sub with a name that been when it when through the loop
    Image.fromarray(img).save(
        OUTPUT_DIR / f"cluster_{label}" / f"img_{idx}.png"
    )

# this was used when desiding on what k value to use to look at the graph and looking
# at the elbow

# inertias = []
# k_values = range(1, 300)

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


