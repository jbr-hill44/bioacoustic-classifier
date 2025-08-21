import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from keras import layers
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import src.data_processing.data_augmentation as daug
from importlib import reload
import src.models.CNN as CNN
reload(CNN)
import numpy as np
from src.models.active_learning_kcluster import kCenterGreedy


df = pd.read_csv("/Users/jameshill/PycharmProjects/bioacoustic-classifier/src/data/annotations/spectrogram_labels.csv")
df['filepath'] = "/Users/jameshill/PycharmProjects/bioacoustic-classifier/data/processed/spectrogram_3s/" + df['filename'] + ".png"
df['split_labels'] = df['label'].str.split('_and_')

# Get arrays
filepaths = df['filepath'].values
# Initialise and fit multi-label encoder
mle = MultiLabelBinarizer()
multi_labels = mle.fit_transform(df['split_labels'])
labels = multi_labels

classes = list(mle.classes_)
target = 'eurasian_skylark'
bg = 'background_noise'

t_idx = classes.index(target)
bg_idx = classes.index(bg)

Y = labels  # (N, C)

# positives: target present
pos_mask = (Y[:, t_idx] == 1)

# clean negatives: background only (no other labels)
bg_only_mask = (Y[:, bg_idx] == 1) & (Y.sum(axis=1) == 1)

idx_bin = np.where(pos_mask | bg_only_mask)[0]
y_bin   = (Y[idx_bin, t_idx] == 1).astype('float32')  # 1=target, 0=background-only

print("Binary set size:", idx_bin.size,
      "| positives:", y_bin.sum(),
      "| negatives:", (1 - y_bin).sum())


from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1929)
train_idx_bin, val_idx_bin = next(sss.split(idx_bin, y_bin))
train_idx = idx_bin[train_idx_bin]
val_idx   = idx_bin[val_idx_bin]

print("Train/Val:", train_idx.size, val_idx.size)
print("Pos ratio train:", y_bin[train_idx_bin].mean(),
      "| val:", y_bin[val_idx_bin].mean())

def make_binary_ds(paths, y_bin, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((paths, y_bin.astype('float32').reshape(-1, 1)))
    ds = ds.map(daug.decode_image, num_parallel_calls=tf.data.AUTOTUNE)  # decode_image returns (image, label)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_binary_ds(filepaths[train_idx], y_bin[train_idx_bin])
val_ds   = make_binary_ds(filepaths[val_idx],   y_bin[val_idx_bin])

def define_binary_cnn(input_shape=(64,512,1)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      activation="relu",
                      padding="same",
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

bin_model = define_binary_cnn()
bin_model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[keras.metrics.AUC(curve='PR', name='auc_pr'),
                           keras.metrics.AUC(curve='ROC', name='auc_roc')])

hist = bin_model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=1)
print("Final val AUC-PR:", hist.history['val_auc_pr'][-1])
print("Final val AUC-ROC:", hist.history['val_auc_roc'][-1])

from sklearn.metrics import average_precision_score, roc_auc_score

# predict on full val
val_x = tf.data.Dataset.from_tensor_slices(filepaths[val_idx]) \
         .map(lambda p: daug.decode_image(p, tf.zeros([1], tf.float32))[0]) \
         .batch(64)
y_true = y_bin[val_idx_bin]
y_pred = bin_model.predict(val_x, verbose=0).ravel()

print("sklearn val PR-AUC:",  average_precision_score(y_true, y_pred))
print("sklearn val ROC-AUC:", roc_auc_score(y_true, y_pred))


common_classes =
