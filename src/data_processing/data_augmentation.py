import tensorflow as tf
import numpy as np


def upsample_rare(train_labels, train_idx):
    train_idx = np.asarray(train_idx, dtype=int)

    yt = train_labels[train_idx]
    class_freqs = yt.sum(axis=0)
    present = class_freqs > 0
    most_freq = class_freqs[present].max() if present.any() else 1

    upsample_factor = np.ones_like(class_freqs, dtype=float)
    upsample_factor[present] = np.clip(np.ceil(most_freq / class_freqs[present]).astype(int), 1, 5)
    upsample_factor = upsample_factor.astype(int)

    classes_upsampled = yt * upsample_factor
    label_upsample_factor = classes_upsampled.max(axis=1).astype(int)
    extra_counts = np.maximum(label_upsample_factor - 1, 0).astype(int)

    base_idx = train_idx
    dup_idx = np.repeat(train_idx, extra_counts)
    all_idx = np.concatenate([base_idx, dup_idx])
    all_flag = (np.concatenate([np.zeros(len(base_idx), bool),
                                np.ones(len(dup_idx), bool)])
                if dup_idx.size else np.zeros(len(base_idx), bool))

    rng = np.random.default_rng(1929)
    perm = rng.permutation(all_idx.shape[0])
    all_idx = all_idx[perm]
    all_flag = all_flag[perm]
    return all_idx, all_flag

# Define data augmentation functions

def vertical_roll(image):
    shift = tf.random.uniform(shape=[], minval=-5, maxval=6, dtype=tf.int32)
    return tf.roll(image, shift=shift, axis=0)


def horizontal_roll(image):
    shift = tf.random.uniform(shape=[], minval=-50, maxval=51, dtype=tf.int32)
    return tf.roll(image, shift=shift, axis=1)


def warp(image):
    angle = tf.random.uniform([], -0.05, 0.05)  # radians
    rotated = tf.image.rotate(image, angles=angle, fill_mode="constant")
    return rotated


def add_noise(image):
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.02)
    noised = tf.clip_by_value(image + noise, 0.0, 1.0)
    return noised


# This is to augment the image by 3 of the 4 possible
def augment_k_of_n(image, label, k=3):
    ops = [vertical_roll, horizontal_roll, add_noise]
    idx = tf.range(len(ops))
    idx = tf.random.shuffle(idx)[:k]

    def apply_op(im, op):
        return tf.switch_case(op, branch_fns=[
            lambda: vertical_roll(im),
            lambda: horizontal_roll(im),
            #lambda: warp(im),
            lambda: add_noise(im)
        ])

    for op_idx in tf.unstack(idx):
        image = apply_op(image, op_idx)
    return image, label


def decode_image(filename, label):
    # read in and process image
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [64, 512])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def decode_image_with_aug(filename, label, do_aug):
    # read in and process image
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [64, 512])
    image = tf.cast(image, tf.float32) / 255.0

    # process if augment
    def aug():
        img, lbl = augment_k_of_n(image, label)
        return img, lbl

    # process if no augment
    def no_aug():
        return image, label

    # image and label are either aug output or no_aug output, depending on do_aug
    image, label = tf.cond(do_aug, aug, no_aug)
    return image, label