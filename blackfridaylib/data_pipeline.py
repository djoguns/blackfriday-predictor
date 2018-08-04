import os

import tensorflow as tf

from .constants import *

def _parse_csv(rows_string_tensor):
    num_features = len(ALL_FEATS)
    num_columns = num_features + 1
    columns = tf.decode_csv(rows_string_tensor, record_defaults=RECORD_DTYPES)
    features = dict(zip(ALL_FEATS, columns[:num_features]))
    labels = tf.cast(columns[num_features], tf.float32)
    return features, labels

def input_fn(file_name):
    # Extract
    dataset = tf.data.TextLineDataset([file_name])
    dataset = dataset.skip(1)
    dataset = dataset.map(_parse_csv)

    #Transform
    dataset = dataset.batch(batch_size=32)

    # Load
    return dataset

def train_fn():
    return input_fn(TRAIN_FILE)

def eval_fn():
    return input_fn(EVAL_FILE)
