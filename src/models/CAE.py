import tensorflow as tf
from tensorflow import keras
from keras import layers

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

