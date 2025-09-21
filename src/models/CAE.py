import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split, ShuffleSplit
import numpy as np


def unlabelled_decode(filename: tf.Tensor):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [64, 512])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def make_unlabelled_ds(filepaths: np.ndarray, indices: np.ndarray, batch_size: int):
    ds = tf.data.Dataset.from_tensor_slices(filepaths[indices])
    ds = ds.map(unlabelled_decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds

def conv_encoder(input_shape=(64,512,1)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D(2)(x)             # 32x256
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)             # 16x128
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)             # 8x64
    x = layers.Conv2D(256, 3, padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)
     #(8x64x256)
    return keras.Model(inp, x, name="conv_encoder")

def build_cae(input_shape=(64,512,1)):
    enc = conv_encoder(input_shape)
    z   = enc.output                     # (8,64,256)

    x = layers.UpSampling2D((2,2))(z)    # 16x128
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D((2,2))(x)    # 32x256
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D((2,2))(x)    # 64x512
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    out = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)

    cae = keras.Model(enc.input, out, name="cae")
    return cae, enc

# Read in unlabelled data
# These will only be used for pretraining the autoencoder
from pathlib import Path

folder = Path("/Users/jameshill/PycharmProjects/bioacoustic-classifier/data/processed/spectrogram_3s/unlabelled")
files = np.array(sorted(str(p) for p in folder.glob("*.png")), dtype=np.str_)

# Split out some validation data for assessing the autoencoder
ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=1929)
unlabelled_train_idx, unlabelled_val_idx = next(ss.split(files))
cae_train_ds = make_unlabelled_ds(files, unlabelled_train_idx, batch_size=64)
cae_val_ds = make_unlabelled_ds(files, unlabelled_val_idx, batch_size=64)

cae_train_images = cae_train_ds.map(lambda x: (x,x))
cae_val_images = cae_val_ds.map(lambda x: (x,x))

cae, enc = build_cae()
cae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='conv_ae.keras',
        save_best_only=True,
        save_weights_only=False,
        monitor='val_loss')
        ]
# Include early stopping to protect against overfit. Patience is 10 - how many epochs before comparing loss.
early_cb = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                         patience=10, restore_best_weights=True, verbose=1)
# If loss starts to plateau, reduce the LR
plateau_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                               patience=4, min_lr=1e-5, verbose=1)
# Fit the model
cae_history = cae.fit(cae_train_images, epochs=50, validation_data=cae_val_images, callbacks=[callbacks,early_cb,plateau_cb])