import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


def compute_irlbl_and_weights(y_train, power=0.5, eps=1e-6):
    """
    IRLbl[c] = max_pos / pos_c  (Sampaio et al.)
    Convert IRLbl -> positive-class weights via a soft power (default sqrt)
    and normalize to mean 1 so loss scale stays reasonable.
    """
    pos_per_label = y_train.sum(axis=0).astype(float) + eps
    max_pos = pos_per_label.max()
    irlbl = max_pos / pos_per_label
    # turn IRLbl into weights (so rarer labels get larger weights)
    w = np.power(irlbl, power)
    w = w / (w.mean() + eps)  # normalize to mean ~1
    return irlbl, w


def bce_with_positive_weights(alpha_pos):
    """
    alpha_pos: 1D numpy or list of length C (per-class positive weights).
    Assumes model outputs sigmoid probabilities (from_logits=False).
    """
    alpha_pos = tf.constant(alpha_pos, dtype=tf.float32)
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # per-sample, per-class BCE with per-class weight on the positive term
        pos_term = - y_true * tf.math.log(y_pred) * alpha_pos
        neg_term = - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(pos_term + neg_term)
    return loss_fn


def tune_per_class_thresholds(y_true_val, y_prob_val, grid=np.linspace(0.05, 0.95, 19)):
    C = y_true_val.shape[1]
    thresholds = np.full(C, 0.5, dtype=float)
    for c in range(C):
        best_t, best_f1 = 0.5, -1.0
        yt = y_true_val[:, c].astype(int)
        for t in grid:
            yp = (y_prob_val[:, c] >= t).astype(int)
            # if a class is entirely 0/1 in val, f1_score can warn; handle gracefully
            try:
                f1 = f1_score(yt, yp, average='binary', zero_division=0)
            except Exception:
                f1 = 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


def evaluate_f1s(y_true, y_prob, thresholds):
    y_hat = (y_prob >= thresholds[None, :]).astype(int)
    macro_f1 = f1_score(y_true, y_hat, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_hat, average='micro', zero_division=0)
    return macro_f1, micro_f1, y_hat


def compute_scumble_per_instance(Y, irlbl, eps=1e-12):
    """
    SCUMBLE_i = 1 - (G_i / A_i), where
      A_i = arithmetic mean of IRLbl over labels present in instance i
      G_i = geometric mean of IRLbl over labels present in instance i
    For single-label instances, G_i == A_i -> SCUMBLE_i = 0.
    For instances with no positive labels, returns np.nan (ignored in dataset mean).
    """
    N, C = Y.shape
    scumble_i = np.full(N, np.nan, dtype=float)

    # For each instance, get indices of present labels
    for i in range(N):
        idx = np.flatnonzero(Y[i])
        k = idx.size
        if k == 0:
            # undefined (no labels present) -> keep nan
            continue
        vals = irlbl[idx]
        A = vals.mean()
        # geometric mean: exp(mean(log(vals)))
        G = np.exp(np.mean(np.log(np.maximum(vals, eps))))
        scumble_i[i] = 1.0 - (G / (A + eps))
    return scumble_i  # shape: (N,)

def compute_scumble(Y):
    """
    Convenience wrapper: returns
      - irlbl (per-label)
      - scumble_i (per-instance)
      - scumble (dataset mean over defined instances)
      - scumble_cv (coefficient of variation over defined instances)
    """
    irlbl, w = compute_irlbl_and_weights(Y)
    scumble_i = compute_scumble_per_instance(Y, irlbl)
    scumble = np.nanmean(scumble_i)
    # coefficient of variation = std / mean (over valid instances)
    scumble_cv = (np.nanstd(scumble_i) / scumble) if scumble > 0 else np.nan
    return irlbl, w, scumble_i, scumble, scumble_cv

