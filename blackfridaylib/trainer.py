import os
import ast
import json

import tensorflow as tf

from .constants import *
from .utils import latest_timestamp
from .data_pipeline import train_fn, eval_fn

class Crossing:

    def __init__(self, crossing):
        # parse crossing inpt
        self.col1_name = crossing[0]
        self.col2_name = crossing[1]
        self.type = crossing[2]
        self.hash_bucket_size = crossing[3]
        self.dim = None if self.type==IND else crossing[4]

    def make_column(self):
        col1, col2 = FEAT_DEF[self.col1_name], FEAT_DEF[self.col2_name]

        if self.type == IND:
            crossed_column = tf.feature_column.crossed_column([col1, col2], hash_bucket_size=self.hash_bucket_size)
            return tf.feature_column.indicator_column(crossed_column)

        if self.type == EMB:
            return tf.feature_column.embedding_column(
                tf.feature_column.crossed_column([col1, col2], hash_bucket_size=self.hash_bucket_size), self.dim)

    def name_column(self):
        return '{}X{}'.format(self.col1_name, self.col2_name)

def my_rmse(labels, predictions):
    pred_values = predictions['predictions']
    return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}

def design_features(indicator_cols=[], embedding_cols={}, crossings=[]):
    cols = {}

    for col_name in indicator_cols:
        col = FEAT_DEF[col_name]
        cols[col_name] = tf.feature_column.indicator_column(col)

    for col_name, dim in embedding_cols.items():
        col = FEAT_DEF[col_name]
        cols[col_name] = tf.feature_column.embedding_column(col, dim)

    for spec in crossings:
        cross = Crossing(spec)
        crossing_name = cross.name_column()
        cols[crossing_name] = cross.make_column()

    return list(cols.values())

def design_estimator(model_name, feature_cols, est_type, hidden_units=[], deep_feats=None, wide_feats=None):

    if est_type == DNN_REG:
        return tf.estimator.DNNRegressor(
            feature_columns=feature_cols,
            hidden_units=hidden_units,
            model_dir=os.path.join(TENSORBOARD, model_name))

    if est_type == DEEP_WIDE:
        return tf.estimator.DNNLinearCombinedRegressor(
        linear_feature_columns=wide_feats,
        dnn_feature_columns=deep_feats,
        dnn_hidden_units=hidden_units,
        model_dir=os.path.join(TENSORBOARD, model_name)
        )

    raise NameError('Only DNN_REG and DEEP_WIDE estimators are supported')

def define_train_eval_specs(max_steps=DEF_STEPS):
    train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_fn)
    return train_spec, eval_spec

def parse_and_vaildate_args(args):
    # parse
    args.embedding_cols = json.loads(args.embedding_cols)
    args.crossed_cols = [ast.literal_eval(arg) for arg in args.crossed_cols]

    # validate
    if type(args.indicator_cols) != list:
        raise NameError('indicator cols must be a list')
    if len(args.indicator_cols) > MAX_LYRS:
        raise NameError('Cannot exceed {} layers'.format(MAX_LYRS))
    if type(args.embedding_cols) != dict:
        raise NameError('embedding cols must be a dictionary')
    if args.training_steps > MAX_TRAIN_STEPS:
        raise NameError('Cannot specify more than {} training steps'.format(MAX_TRAIN_STEPS))

    return args

def main(args):
    params = parse_and_vaildate_args(args)
    fc = design_features(indicator_cols=params.indicator_cols, embedding_cols=params.embedding_cols, crossings=params.crossed_cols)

    estimator = design_estimator(params.model_name, fc, params.estimator_type, params.hidden_units)
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)

    # train model
    train_spec, eval_spec = define_train_eval_specs(max_steps=params.training_steps)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # save model
    feature_spec = tf.feature_column.make_parse_example_spec(fc)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_savedmodel(os.path.join(SAVEDMODEL_DIR, params.model_name), serving_input_receiver_fn)

