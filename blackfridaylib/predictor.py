import os

import numpy as np
import pandas as pd
import tensorflow as tf

from .constants import *
from .utils import latest_timestamp


def parse_features(record):
    return {
    AGE : tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[record[AGE].encode('utf-8')])),

    GENDER: tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[record[GENDER].encode('utf-8')])),

    OCC: tf.train.Feature(
        int64_list=tf.train.Int64List(value=[record[OCC]])),

    MS: tf.train.Feature(
        int64_list=tf.train.Int64List(value=[record[MS]])),

    CITY: tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[record[CITY].encode('utf-8')])),

    STAY: tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[record[STAY].encode('utf-8')])),

    PROD1: tf.train.Feature(
        int64_list=tf.train.Int64List(value=[record[PROD1]])),

    PROD2: tf.train.Feature(
        int64_list=tf.train.Int64List(value=[record[PROD2]])),

    PROD3: tf.train.Feature(
        int64_list=tf.train.Int64List(value=[record[PROD3]])),

    PRODUCT_ID: tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[record[PRODUCT_ID].encode('utf-8')]))
    }

def parse_csv(fname):
    df = pd.read_csv(fname)
    return df.to_dict(orient='records')

def parse_records(strings):
    records = []
    for string in strings:
        pass

def load_examples(data):
    examples = []
    for record in data:
        features = parse_features(record)
        example = tf.train.Example(features=tf.train.Features(feature=features))
        examples.append(example)
    return examples

def load_estimator(full_model_dir):
    latest_version = latest_timestamp(full_model_dir)
    path = os.path.join(full_model_dir, latest_version)
    return tf.contrib.predictor.from_saved_model(path)

def write_predictions(predictions, file='predictions.csv'):
    with open(file, 'w+') as f:
        for prediction in predictions:
            f.write("{}\n".format(str(prediction[0])))

def main(params):
    # where to look for the test data
    # can take file, or single record
    if params.fpath:
        data = parse_csv(params.fpath)
    else:
        data = parse_records(params.record)

    examples = load_examples(data)
    model_inputs = [example.SerializeToString() for example in examples]

    estimator = load_estimator(params.model_path)
    output_dict = estimator({"inputs":model_inputs})
    predictions = output_dict['outputs']
    if params.fpath:
        write_predictions(predictions, file=params.outfile)
    else:
        print([p[0] for p in predictions])

if __name__ == '__main__':
    main()
