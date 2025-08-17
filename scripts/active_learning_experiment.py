import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from src.data_processing.data_augmentation import upsample_rare, decode_image, decode_image_with_aug
from src.models.CNN import define_cnn
from src.models.active_learning_kcluster import kCenterGreedy

# Read in data
df = pd.read_csv("/Users/jameshill/PycharmProjects/bioacoustic-classifier/src/data/annotations/spectrogram_labels.csv")
df['filepath'] = "/Users/jameshill/PycharmProjects/bioacoustic-classifier/data/processed/spectrogram_3s/" + df['filename'] + ".png"
df['split_labels'] = df['label'].str.split('_and_')

# Get arrays
filepaths = df['filepath'].values

# Initialise and fit multi-label encoder
mle = MultiLabelBinarizer()
multi_labels = mle.fit_transform(df['split_labels'])
labels = multi_labels

cnn = define_cnn(mle.classes_)

def make_train_ds(filepaths, labels, indices, flags, batch_size=32, seed=1929):
    idx = np.asarray(indices, dtype=int)
    labs = np.asarray(labels)[idx].astype('float32')
    paths = np.asarray(filepaths)[idx]
    flgs = np.asarray(flags).astype('bool')

    ds = tf.data.Dataset.from_tensor_slices((paths, labs, flgs))
    ds = ds.shuffle(len(idx), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(decode_image_with_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_eval_ds(filepaths, labels, indices, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((filepaths[indices],
                                             labels[indices].astype('float32')))
    ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds



# Test/ train split
# Extract indices of data, test/ train split
# This is so images themselves do not need duplicating
# but instead augmentation will be applied when relevant index occurs
idx = np.arange(len(filepaths))
train_idx, test_val_idx = train_test_split(idx, test_size=0.3, random_state=1929, shuffle=True)
test_idx, val_idx = train_test_split(test_val_idx, test_size=0.8, random_state=1929, shuffle=True)

rng = np.random.default_rng(1929)
m0 = 400  # images in initial training round
b = 100  # samples per batch
rounds = 8  # number of rounds

L0 = rng.choice(train_idx, m0, replace=False)  # initial labelled subset
U0 = np.setdiff1d(train_idx, L0, assume_unique=False)  # remaining 'unlabelled' pool


def create_training_set(model, train_idx, val_idx, epochs=10, batch_size=32):
    # Upsample only on training set
    epoch_idx, epoch_flag = upsample_rare(train_labels=labels, train_idx=train_idx)
    # These handle processing of images
    train_ds = make_train_ds(filepaths, labels, epoch_idx, epoch_flag, batch_size=batch_size)
    val_ds = make_eval_ds(filepaths, labels, val_idx, batch_size=batch_size)

    # Compile model
    num_classes = labels.shape[1]
    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[
                      keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=num_classes),
                      keras.metrics.AUC(curve='PR', multi_label=True, num_labels=num_classes)
                  ])
    # Train model on training data
    callbacks = keras.callbacks.EarlyStopping(monitor='val_auc_pr', mode='max', patience=2, restore_best_weights=True)
    base_model = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0
    )
    return base_model

def get_embeddings(model, pool_idx, batch_size=32):
    # model’s penultimate layer (Dense(128))
    embed_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    pool_ds = make_eval_ds(filepaths, labels, pool_idx, batch_size)  # no aug
    E = embed_model.predict(pool_ds, verbose=0)  # shape: (len(pool_idx), 128)
    return E

# kCenterGreedy was built to expect scikit-learn models, which have a transform(x) method
# We have embeddings as per last step and do not need to manually map them
# Create identity model to imitate behaviour of the transform method
class IdentityModel:
    # provides a .transform to satisfy kCenterGreedy API
    def __init__(self, feats):
        self._feats = feats

    def transform(self, X):
        return self._feats


def select_kcenter_batch(embeddings, already_selected_local, b, metric='euclidean'):
    # embeddings: (n_pool, d)
    dummy_X = embeddings  # X argument unused except for shape; transform() returns feats
    sampler = kCenterGreedy(X=dummy_X, y=None, seed=1929, metric=metric)
    sampler.features = embeddings
    # already_selected_local: indices into *pool array*, not global ids
    batch_local = sampler.select_batch_(model=IdentityModel(embeddings),
                                        already_selected=already_selected_local, N=b)
    return np.array(batch_local, dtype=int)


def run_al_experiment(L_init, U_init, rounds, b, seed=1929):
    rng = np.random.default_rng(seed)

    # containers
    results = {
        'round': [],
        'method': [],
        'n_sampled': [],
        'val_auc_roc': [],
        'test_auc_roc': []
    }

    # fixed eval sets
    val_ds = make_eval_ds(filepaths, labels, val_idx, batch_size=64)
    test_ds = make_eval_ds(filepaths, labels, test_idx, batch_size=64)
    # two tracks: active and random
    for method in ['active', 'random']:
        L = L_init.copy()
        U = U_init.copy()
        # fresh model per method (fair comparison)
        model = define_cnn(mle.classes_)
        # round 0: train on initial L
        hist = create_training_set(model, L, val_idx)
        # evaluate
        val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
        test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
        results['round'].append(0)
        results['method'].append(method)
        results['n_sampled'].append(len(L))
        results['val_auc_roc'].append(val_metrics['auc'])
        results['test_auc_roc'].append(test_metrics['auc'])

        for r in range(1, rounds+1):
            # select b new labels from U
            if method == 'active':
                # get embeddings for U
                E = get_embeddings(model, U)
                # already_selected indices relative to U (none, we’re selecting new each round)
                selected_local = select_kcenter_batch(E, already_selected_local=[], b=b)
                newly_selected = U[selected_local]
            else:
                newly_selected = rng.choice(U, size=b, replace=False)
            # "oracle": use ground-truth labels
            L = np.concatenate([L, newly_selected])
            U = np.setdiff1d(U, newly_selected, assume_unique=False)

            # retrain (or continue training) on enlarged L
            hist = create_training_set(model, L, val_idx, epochs=8)
            val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
            test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
            results['round'].append(r)
            results['method'].append(method)
            results['n_sampled'].append(len(L))
            results['val_auc_roc'].append(val_metrics['auc'])
            results['test_auc_roc'].append(test_metrics['auc'])
    # to DataFrame for plotting/stats
    out = pd.DataFrame(results)
    return out


