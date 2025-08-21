import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import src.data_processing.data_augmentation as daug
from importlib import reload
import src.models.CNN as CNN
reload(CNN)
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
cnn = CNN.define_cnn(mle.classes_)
assert cnn.output_shape[-1] == len(mle.classes_)
def make_train_ds(filepaths, labels, indices, flags, batch_size=32, seed=1929):
    idx = np.asarray(indices, dtype=int)
    labs = np.asarray(labels)[idx].astype('float32')
    paths = np.asarray(filepaths)[idx]
    flgs = np.asarray(flags).astype('bool')

    ds = tf.data.Dataset.from_tensor_slices((paths, labs, flgs))
    ds = ds.shuffle(len(idx), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(daug.decode_image_with_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_eval_ds(filepaths, labels, indices, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((filepaths[indices],
                                             labels[indices].astype('float32')))
    ds = ds.map(daug.decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds



# Test/ train split
# Extract indices of data, test/ train split
# This is so images themselves do not need duplicating
# but instead augmentation will be applied when relevant index occurs
idx = np.arange(len(filepaths))
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1929)
rest_idx, test_idx = next(msss.split(idx, labels))
train_idx, val_idx = next(msss.split(rest_idx, labels[rest_idx]))
# train_idx, test_val_idx = iterative_train_test_split(idx, test_size=0.3, random_state=1929, shuffle=True)
# val_idx, test_idx = iterative_train_test_split(test_val_idx, test_size=0.8, random_state=1929, shuffle=True)

rng = np.random.default_rng(1929)
m0 = 400  # images in initial training round
b = 100  # samples per batch
rounds = 8  # number of rounds

L0 = rng.choice(train_idx, m0, replace=False)  # initial labelled subset
U0 = np.setdiff1d(train_idx, L0, assume_unique=False)  # remaining 'unlabelled' pool

def create_training_set(model, train_idx, val_idx, epochs=10, batch_size=32):
    # Upsample only on training set
    # epoch_idx, epoch_flag = daug.upsample_rare(train_labels=labels, train_idx=train_idx)
    # These handle processing of images
    train_ds = make_eval_ds(filepaths, labels, train_idx, batch_size=batch_size) # change to make_train_ds if you want to use augmentation
    val_ds = make_eval_ds(filepaths, labels, val_idx, batch_size=batch_size)

    # y_val = labels[val_idx]
    # valid_mask = ((y_val.sum(axis=0) > 0) & ((y_val == 0).sum(axis=0) > 0)).astype('float32')

    # Compile model
    num_classes = labels.shape[1]
    pr = keras.metrics.AUC(
        name='auc_pr', curve='PR', multi_label=True, num_labels=num_classes
    )

    model.compile(
        optimizer='adam',
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[pr]
    )
    # Train model on training data
    callbacks = keras.callbacks.EarlyStopping(
        monitor='val_auc_pr', mode='max', patience=2, restore_best_weights=True
    )
    base_model = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0
    )
    return base_model

def make_input_ds(filepaths, indices, batch_size=64):
    paths = np.asarray(filepaths)[np.asarray(indices, dtype=int)]
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: daug.decode_image(p, tf.zeros([labels.shape[1]], tf.float32))[0],  # take only image
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

epoch_idx, epoch_flag = daug.upsample_rare(train_labels=labels, train_idx=train_idx)
train_ds = make_train_ds(filepaths, labels, epoch_idx, epoch_flag, batch_size=32)
model = CNN.define_cnn(mle.classes_)
model_all = create_training_set(model=model,
                                train_idx=train_idx,
                                val_idx=val_idx,
                                epochs=20)


auc_pr = model_all.history['auc_pr']
val_auc_pr = model_all.history['val_auc_pr']
epochs = range(1, len(auc_pr) + 1)

plt.plot(epochs, auc_pr, 'bo', label="Training AUC-PR")
plt.plot(epochs, val_auc_pr, 'b', label="Validation AUC-PR")
plt.title("Training and Validation AUC-PR")
plt.legend()
plt.show()

y_true = labels[val_idx].astype('float32')
y_pred = model.predict(make_input_ds(filepaths, val_idx), verbose=0)

# Restrict to classes with both pos & neg in val (optional but safer)
valid = (y_true.sum(axis=0) > 0) & ((y_true == 0).sum(axis=0) > 0)
yt, yp = y_true[:, valid], y_pred[:, valid]

# PR-AUCs
ap_macro = average_precision_score(yt, yp, average='macro')
ap_weighted = average_precision_score(yt, yp, average='weighted')  # weights by class support
ap_micro = average_precision_score(yt, yp, average='micro')
print(f"AP macro={ap_macro:.3f} | AP weighted={ap_weighted:.3f} | AP micro={ap_micro:.3f}")

# Hamming loss metrics
threshold = 0.4  # tune this
yp_binary = (yp >= threshold).astype(int)
from sklearn.metrics import hamming_loss
hamming = hamming_loss(yt, yp_binary)
print(f"Hamming Loss={hamming:.3f}")


import numpy as np
from sklearn.metrics import average_precision_score

# yt, yp from your last cell (restricted to valid classes)
nC = yt.shape[1]

per_ap = np.zeros(nC)
pos_pred_rate_05 = np.zeros(nC)   # fraction predicted positive at 0.5
mean_prob = np.zeros(nC)          # average predicted probability
best_f1 = np.zeros(nC)
best_thr = np.zeros(nC)

for c in range(nC):
    y, p = yt[:, c], yp[:, c]  # take all rows, for each column (class)
    per_ap[c] = average_precision_score(y, p)
    mean_prob[c] = float(p.mean())

    # default 0.5 threshold: how often does model predict positive?
    pos_pred_rate_05[c] = float((p >= 0.5).mean())

# Map these back to class names you evaluated (the 'valid' subset)
valid_classes = np.array(list(mle.classes_))[valid]
summary = np.c_[per_ap, mean_prob, pos_pred_rate_05, best_f1, best_thr]
idx_sorted = np.argsort(-per_ap)  # best first
for i in idx_sorted[:nC]:
    print(f"{valid_classes[i]:30s} AP={per_ap[i]:.3f}  mean_p={mean_prob[i]:.3f}  "
          f"pos@0.5={pos_pred_rate_05[i]:.3f}  bestF1={best_f1[i]:.3f}@thr={best_thr[i]:.2f}")


hamming = np.zeros(nC)
for c in range(nC):
    y, p = yt[:, c], yp[:, c]  # take all rows, for each column (class)
    hamming[c] = hamming_loss(y, (p >= threshold).astype(int))

# Map these back to class names you evaluated (the 'valid' subset)
valid_classes = np.array(list(mle.classes_))[valid]
summary = np.c_[hamming]
for i in idx_sorted[:nC]:
    print(f"{valid_classes[i]:30s}" 
          f"Hamming={hamming[i]}")

def get_embeddings(model, pool_idx, batch_size=64):
    pool_idx = np.asarray(pool_idx, dtype=int)
    if pool_idx.size == 0:
        # return a correctly-shaped empty array
        embed_dim = model.layers[-2].output_shape[-1]
        return np.empty((0, embed_dim), dtype=np.float32)

    embed_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    pool_x = make_input_ds(filepaths, pool_idx, batch_size=batch_size)  # IMAGES ONLY
    return embed_model.predict(pool_x, verbose=0)


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
        'val_auc_pr': [],
        'test_auc_roc': [],
        'test_auc_pr': []
    }

    # fixed eval sets
    val_ds = make_eval_ds(filepaths, labels, val_idx, batch_size=64)
    test_ds = make_eval_ds(filepaths, labels, test_idx, batch_size=64)
    # two tracks: active and random
    for method in ['active', 'random']:
        L = L_init.copy()
        U = U_init.copy()
        # fresh model per method (fair comparison)
        model = CNN.define_cnn(mle.classes_)
        # round 0: train on initial L
        hist = create_training_set(model, L, val_idx)
        # evaluate
        val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
        test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
        results['round'].append(0)
        results['method'].append(method)
        results['n_sampled'].append(len(L))
        results['val_auc_roc'].append(val_metrics['auc_roc'])
        results['val_auc_pr'].append(val_metrics['auc_pr'])
        results['test_auc_roc'].append(test_metrics['auc_roc'])
        results['test_auc_pr'].append(test_metrics['auc_pr'])

        for r in range(1, rounds+1):
            if len(U) == 0:
                print(f"[{method}] Pool empty at round {r}. Stopping.")
                break

            b_round = min(b, len(U))
            # select b new labels from U
            if method == 'active':
                # get embeddings for U
                E = get_embeddings(model, U)
                if E.shape[0] == 0:
                    print(f"[{method}] No pool items to embed at round {r}.")
                    break

                print(f"[{method}] round {r} | |U|={len(U)} | b={b_round}")
                selected_local = select_kcenter_batch(E, already_selected_local=[], b=b_round)
                newly_selected = U[selected_local]
            else:
                print(f"[{method}] round {r} | |U|={len(U)} | b={b_round}")
                newly_selected = rng.choice(U, size=b_round, replace=False)
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
            results['val_auc_roc'].append(val_metrics['auc_roc'])
            results['val_auc_pr'].append(val_metrics['auc_pr'])
            results['test_auc_roc'].append(test_metrics['auc_roc'])
            results['test_auc_pr'].append(test_metrics['auc_pr'])
    # to DataFrame for plotting/stats
    out = pd.DataFrame(results)
    return out


# df_res = run_al_experiment(L0, U0, rounds, b)
#
# plt.figure(figsize=(7, 4))
# for method in ['active', 'random']:
#     d = df_res[df_res['method'] == method].sort_values('n_sampled')
#     plt.plot(d['n_sampled'], d['val_auc_roc'], marker='o', label=method)
# plt.xlabel('# labeled samples')
# plt.ylabel('Val AUC-ROC')
# plt.title('Learning curves')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()



