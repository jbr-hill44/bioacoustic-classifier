# This script runs the experiments.
# It will pull and perform the test/ train split on the labelled data.
# It then performs the active learning experiment, with and without pretraining.
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import recall_score, f1_score, precision_score, hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import src.data_processing.data_augmentation as daug
import src.data_processing.imbalance_metrics as imb
from importlib import reload
import src.models.CNN as CNN
reload(CNN)
from src.models.active_learning_kcluster import kCenterGreedy
import gdown
import os

# Read in data
df = pd.read_csv("/Users/jameshill/PycharmProjects/bioacoustic-classifier/src/data/annotations/spectrogram_labels.csv")
df['filepath'] = "/Users/jameshill/PycharmProjects/bioacoustic-classifier/data/processed/spectrogram_3s/" + df['filename'] + ".png"
df['split_labels'] = df['label'].str.split('_and_')

# Load Autoencoder model from Google Drive
os.makedirs("src/models", exist_ok=True)
url = "https://drive.google.com/file/d/1XWiqBNZS5PpldarwKCfspGk3PRWk_akH/view?usp=drive_link"
gdown.download(url=url, output="src/models/encoder_final.keras", fuzzy=True)

# PARAMETERS
rng = np.random.default_rng(1929)
m0 = 100  # images in initial training round
b = 20  # samples per batch
rounds = 45  # number of rounds
TARGET_TEST = 243
TARGET_VAL = 200

# Get arrays
filepaths = df['filepath'].values
# Initialise and fit multi-label encoder
mle = MultiLabelBinarizer()
# Create multi-hot encodings
multi_labels = mle.fit_transform(df['split_labels'])
labels = multi_labels

# Define classification head for loaded encoder.
# This works as the encoder has been defined to match the convolutional layers of the CNN
def build_classifier_from_encoder(encoder_path: str, classes, input_shape=(64,512,1), freeze_encoder: bool = True):
    num_classes = len(classes)

    encoder = keras.models.load_model(encoder_path)
    encoder.trainable = not freeze_encoder

    inputs = keras.Input(shape=input_shape)
    z = encoder(inputs)
    z = keras.layers.GlobalAveragePooling2D()(z)
    z = keras.layers.Dropout(0.5)(z)
    z = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4),
                           name='embed_dense')(z)
    z = keras.layers.Dropout(0.4)(z)
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(z)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# Only needed if using data augmentation
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

# Main function for sourcing tensors from images.
# Uses filepaths to locate images with specified indices, creates tensors from them
# Returns tensors in batches of specified batch size
def make_ds(filepaths, labels, indices, batch_size, shuffle=False, seed = 1929):
    ds = tf.data.Dataset.from_tensor_slices((filepaths[indices],
                                             labels[indices].astype('float32')))
    if shuffle:
        ds = ds.shuffle(len(indices), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(daug.decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# Test/ train split
# Extract indices of data, test/ train split
# This is so images themselves do not need duplicating
# but instead augmentation will be applied when relevant index occurs
idx = np.arange(len(filepaths))
frac_test = TARGET_TEST / len(filepaths)
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=frac_test, random_state=1929)
rest_pos, test_pos = next(msss.split(idx, labels))
rest_idx = idx[rest_pos]
test_idx = idx[test_pos]  # Important: This test set will be used in the pretrained version also

msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=200, random_state=1929)
train_pos, val_pos = next(msss_val.split(rest_idx, labels[rest_idx]))   # positions into rest_idx
train_idx = rest_idx[train_pos]   # map back to original indices
val_idx = rest_idx[val_pos]



L0 = rng.choice(train_idx, m0, replace=False)  # initial labelled subset
U0 = np.setdiff1d(train_idx, L0, assume_unique=False)  # remaining 'unlabelled' pool

def create_training_set(model, train_idx, val_idx, epochs=10, batch_size=64):

    # These handle processing of images
    train_ds = make_ds(filepaths, labels, train_idx, batch_size=batch_size, shuffle=True)
    val_ds = make_ds(filepaths, labels, val_idx, batch_size=batch_size)

    # Get labels for training set and validation set
    y_train = labels[train_idx].astype(np.float32)
    y_val = labels[val_idx].astype(np.float32)
    num_classes = y_train.shape[1]

    # Create irlbl values and associated weights for weighting and tuning
    train_irlbl, irlbl_weights = imb.compute_irlbl_and_weights(y_train, power=0.5)
    irlbl_weights = np.minimum(irlbl_weights, 10.0)

    # Define training metric - what we'll track during training
    pr = keras.metrics.AUC(
        name='auc_pr', curve='PR', multi_label=True, num_labels=num_classes
    )

    # Compile model
    model.compile(
        optimizer='adam',
        loss=imb.bce_with_positive_weights(irlbl_weights),
        metrics=[pr]
    )
    # Train model on training data
    es = keras.callbacks.EarlyStopping(monitor='val_auc_pr', mode='max',
                                       patience=3, restore_best_weights=True)

    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)

    y_prob_val = model.predict(val_ds, verbose=0)
    thresholds = imb.tune_per_class_thresholds(y_true_val=y_val, y_prob_val=y_prob_val)

    # metrics on Val with tuned thresholds
    macro_f1, micro_f1, _ = imb.evaluate_f1s(y_val, y_prob_val, thresholds)
    print("y_prob_val stats: min", y_prob_val.min(), "max", y_prob_val.max(),
          "mean", y_prob_val.mean(), flush=True)
    # Print out macro and micro f1 scores for if the threshold was fixed at 0.5
    macro_f1_fixed, micro_f1_fixed, _ = imb.evaluate_f1s(y_val, y_prob_val,
                                                         np.full(y_val.shape[1], 0.5))
    print("Fixed-0.5 F1 macro/micro:", macro_f1_fixed, micro_f1_fixed)

    # how many probs exceed the tuned thresholds?
    y_hat_val = (y_prob_val >= thresholds[None, :]).astype(int)
    print("Any positives predicted on val? ", y_hat_val.sum() > 0,
          "| total positives predicted:", y_hat_val.sum(), flush=True)

    # per-class val positives (ground truth) — if many are zero, macro-F1 will be fragile
    per_class_val_pos = y_val.sum(axis=0)
    print("Val classes with no positives:", int((per_class_val_pos == 0).sum()), flush=True)

    # thresholds sanity
    print("thresholds summary: min", thresholds.min(), "max", thresholds.max(), flush=True)

    return model, thresholds, {"val_macro_f1": macro_f1, "val_micro_f1": micro_f1, "history": history}


def get_embeddings(model, pool_idx, batch_size=64):
    pool_idx = np.asarray(pool_idx, dtype=int)
    if pool_idx.size == 0:
        # return a correctly-shaped empty array
        embed_dim = model.get_layer('embed_dense').output_shape[-1]
        return np.empty((0, embed_dim), dtype=np.float32)

    embed_model = keras.Model(inputs=model.input, outputs=model.get_layer('embed_dense').output)
    pool_x = make_input_ds(filepaths, pool_idx, batch_size=batch_size)  # IMAGES ONLY
    return embed_model.predict(pool_x, verbose=0)

# We do not want the exact same instance of either model used for both active and random conditions
# Define model factories to return new models
def fresh_baseline_model():
    # brand-new scratch CNN each time this is called
    return CNN.define_cnn(mle.classes_)

def fresh_pretrained_model():
    # brand-new encoder+head each time this is called
    return build_classifier_from_encoder("src/models/encoder_final.keras", mle.classes_, freeze_encoder=False)

# kCenterGreedy was built to expect scikit-learn models, which have a transform(x) method
# We have embeddings as per get_embeddings and do not need to manually map them
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


def make_input_ds(filepaths, indices, batch_size=64):
    paths = np.asarray(filepaths)[np.asarray(indices, dtype=int)]
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: daug.decode_image(p, tf.zeros([labels.shape[1]], tf.float32))[0],  # take only image
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def run_al_experiment(L_init, U_init, rounds, b, model_factory, seed=1929, batch_size=64, epochs=10):
    rng = np.random.default_rng(seed)

    # containers
    results = {
        'round': [],
        'method': [],
        'n_sampled': [],
        'val_macro_f1': [],
        'val_micro_f1': []
    }
    final_models = {}
    final_thresholds = {}

    # two tracks: active and random
    for method in ['active', 'random']:
        L = L_init.copy()
        U = U_init.copy()

        # fresh model per method (fair comparison)
        cnn_model = model_factory()
        # round 0: train on initial L
        # 1) train on L, tune thresholds on V
        model, thresholds, metrics = create_training_set(
            model=cnn_model, train_idx=L, val_idx=val_idx, batch_size=batch_size, epochs=epochs
        )
        # evaluate
        print(f"[Val] F1 macro: {metrics['val_macro_f1']:.3f} | [Val] F1 micro: {metrics['val_micro_f1']:.3f}", flush=True)
        results['round'].append(0)
        results['method'].append(method)
        results['n_sampled'].append(len(L))
        results['val_macro_f1'].append(metrics['val_macro_f1'])
        results['val_micro_f1'].append(metrics['val_micro_f1'])

        for r in range(1, rounds+1):
            print(f"\n=== Round {r + 1}/{rounds} | Method: {method} | L={len(L)} | U={len(U)} ===", flush=True)

            if (r == rounds - 1) or (len(U) == 0):
                print(f"[{method}] Pool empty at round {r}. Stopping.", flush=True)
                break

            b_round = min(b, len(U))
            # select b new labels from U
            if method == 'active':
                # get embeddings for U
                E = get_embeddings(model, U)
                if E.shape[0] == 0:
                    print(f"[{method}] No pool items to embed at round {r}.", flush=True)
                    break

                print(f"[{method}] round {r} | |U|={len(U)} | b={b_round}", flush=True)
                selected_local = select_kcenter_batch(E, already_selected_local=[], b=b_round)
                newly_selected = U[selected_local]
            else:
                print(f"[{method}] round {r} | |U|={len(U)} | b={b_round}", flush=True)
                newly_selected = rng.choice(U, size=b_round, replace=False)
            # "oracle": use ground-truth labels
            L = np.concatenate([L, newly_selected])
            U = np.setdiff1d(U, newly_selected, assume_unique=False)

            # retrain (or continue training) on enlarged L
            model, thresholds, metrics = create_training_set(
                model=cnn_model, train_idx=L, val_idx=val_idx, batch_size=batch_size, epochs=epochs
            )
            results['round'].append(r)
            results['method'].append(method)
            results['n_sampled'].append(len(L))
            results['val_macro_f1'].append(metrics['val_macro_f1'])
            results['val_micro_f1'].append(metrics['val_micro_f1'])
        final_models[method] = model
        final_thresholds[method] = thresholds
    # to DataFrame for plotting/stats
    out = pd.DataFrame(results)
    return out, final_models, final_thresholds

# Pretraining
df_pretraining, final_models_pretraining, final_thresholds_pretraining = run_al_experiment(
    L_init=L0, U_init=U0, rounds=rounds, b=b, model_factory=fresh_pretrained_model
)

# No pretraining
df_no_pretraining, final_models_no_pretraining, final_thresholds_no_pretraining = run_al_experiment(
    L_init=L0, U_init=U0, rounds=rounds, b=b, model_factory=fresh_baseline_model
)

# Evaluation/ examining performance
def plot_learning_curves(df, metric, title):
    plt.figure(figsize=(7,4))
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("n_sampled")
        plt.plot(sub["n_sampled"], sub[metric], marker="o", label=method.capitalize())
    plt.xlabel("# labelled samples (|L|)")
    plt.ylabel(metric.replace("_"," ").title())
    plt.title(title); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

plot_learning_curves(df_no_pretraining, 'val_macro_f1', 'Macro F-1 on Fixed Validation Set')
plot_learning_curves(df_no_pretraining, 'val_micro_f1', 'Micro F-1 on Fixed Validation Set')

def f1_eval_on_test(model):
    y_true = labels[test_idx].astype(int)

    no_pre = final_models_no_pretraining[model]
    pre = final_models_pretraining[model]

    no_pre_thresh = final_thresholds_no_pretraining[model]
    pre_thresh = final_thresholds_pretraining[model]

    y_pred_no_pre = no_pre.predict(make_input_ds(filepaths, test_idx), verbose=0)
    y_pred_pre = pre.predict(make_input_ds(filepaths, test_idx), verbose=0)

    y_bin_no_pre = (y_pred_no_pre >= no_pre_thresh).astype(int)
    y_bin_pre = (y_pred_pre >= pre_thresh).astype(int)

    f1_micro_no_pretrain = f1_score(y_true, y_bin_no_pre, labels=np.arange(labels.shape[1]), average='micro')
    f1_macro_no_pretrain = f1_score(y_true, y_bin_no_pre, labels=np.arange(labels.shape[1]), average='macro')
    f1_micro_pretrain = f1_score(y_true, y_bin_pre, labels=np.arange(labels.shape[1]), average='micro')
    f1_macro_pretrain = f1_score(y_true, y_bin_pre, labels=np.arange(labels.shape[1]), average='macro')

    return (f"Micro F1 No Pretrain: {f1_micro_no_pretrain} | Macro F1 No Pretrain: {f1_macro_no_pretrain} "
            f"| Micro F1 Pretrain: {f1_micro_pretrain} | Macro F1 Pretrain: {f1_macro_pretrain}")


def recall_eval_on_test(model):
    y_true = labels[test_idx].astype(int)

    no_pre = final_models_no_pretraining[model]
    pre = final_models_pretraining[model]

    no_pre_thresh = final_thresholds_no_pretraining[model]
    pre_thresh = final_thresholds_pretraining[model]

    y_pred_no_pre = no_pre.predict(make_input_ds(filepaths, test_idx), verbose=0)
    y_pred_pre = pre.predict(make_input_ds(filepaths, test_idx), verbose=0)

    y_bin_no_pre = (y_pred_no_pre >= no_pre_thresh).astype(int)
    y_bin_pre = (y_pred_pre >= pre_thresh).astype(int)

    recall_micro_no_pretrain = recall_score(y_true, y_bin_no_pre, labels=np.arange(labels.shape[1]), average='micro')
    recall_macro_no_pretrain = recall_score(y_true, y_bin_no_pre, labels=np.arange(labels.shape[1]), average='macro')
    recall_micro_pretrain = recall_score(y_true, y_bin_pre, labels=np.arange(labels.shape[1]), average='micro')
    recall_macro_pretrain = recall_score(y_true, y_bin_pre, labels=np.arange(labels.shape[1]), average='macro')

    return (f"Micro Recall No Pretrain: {recall_micro_no_pretrain} | Macro Recall No Pretrain: {recall_macro_no_pretrain} "
            f"| Micro Recall Pretrain: {recall_micro_pretrain} | Macro Recall Pretrain: {recall_macro_pretrain}")

recall_eval_on_test('active')

def precision_eval_on_test(model):
    y_true = labels[test_idx].astype(int)

    no_pre = final_models_no_pretraining[model]
    pre = final_models_pretraining[model]

    no_pre_thresh = final_thresholds_no_pretraining[model]
    pre_thresh = final_thresholds_pretraining[model]

    y_pred_no_pre = no_pre.predict(make_input_ds(filepaths, test_idx), verbose=0)
    y_pred_pre = pre.predict(make_input_ds(filepaths, test_idx), verbose=0)

    y_bin_no_pre = (y_pred_no_pre >= no_pre_thresh).astype(int)
    y_bin_pre = (y_pred_pre >= pre_thresh).astype(int)

    prec_micro_no_pretrain = precision_score(y_true, y_bin_no_pre, labels=np.arange(labels.shape[1]), average='micro')
    prec_macro_no_pretrain = precision_score(y_true, y_bin_no_pre, labels=np.arange(labels.shape[1]), average='macro')
    prec_micro_pretrain = precision_score(y_true, y_bin_pre, labels=np.arange(labels.shape[1]), average='micro')
    prec_macro_pretrain = precision_score(y_true, y_bin_pre, labels=np.arange(labels.shape[1]), average='macro')

    return (f"Micro Precision No Pretrain: {prec_micro_no_pretrain} | Macro Precision No Pretrain: {prec_macro_no_pretrain} "
            f"| Micro Precision Pretrain: {prec_micro_pretrain} | Macro Precision Pretrain: {prec_macro_pretrain}")


def hamming_eval_on_test(model):
    y_true = labels[test_idx].astype(int)

    no_pre = final_models_no_pretraining[model]
    pre = final_models_pretraining[model]

    no_pre_thresh = final_thresholds_no_pretraining[model]
    pre_thresh = final_thresholds_pretraining[model]

    y_pred_no_pre = no_pre.predict(make_input_ds(filepaths, test_idx), verbose=0)
    y_pred_pre = pre.predict(make_input_ds(filepaths, test_idx), verbose=0)

    y_bin_no_pre = (y_pred_no_pre >= no_pre_thresh).astype(int)
    y_bin_pre = (y_pred_pre >= pre_thresh).astype(int)

    hamming_no_pretrain = hamming_loss(y_true, y_bin_no_pre)
    hamming_pretrain = hamming_loss(y_true, y_bin_pre)

    return (f"Hamming No Pretrain: {hamming_no_pretrain}"
            f"| Hamming Pretrain: {hamming_pretrain}")

def per_class_metrics (model, pretrain=True):
    if pretrain:
        model = final_models_pretraining[model]
        thresholds = final_thresholds_pretraining[model]
    else:
        model = final_models_no_pretraining[model]
        thresholds = final_thresholds_no_pretraining[model]
    test_ds = make_ds(filepaths, labels, test_idx, batch_size=64)
    y_true_test = labels[test_idx]
    y_prob_test = model.predict(test_ds, verbose=0)
    y_pred_test = (y_prob_test >= thresholds[None, :]).astype(int)
    per_class_f1 = precision_score(y_true_test, y_pred_test, average=None, zero_division=0)
    per_class_f1_dict = dict(zip(mle.classes_, per_class_f1))
    for cls, f1 in per_class_f1_dict.items():
        print(f"{cls}: {f1:.3f}")
