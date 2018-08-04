import argparse

from .constants import *

# General
MDL_NAME = 'model_name'
DEF_MDL_NAME = 'mymodel'

# trainer
TRAINER_DESCRIPTION = ''
IND_COLS = 'indicator_cols'
EMB_COLS = 'embedding_cols'
CROSS_COLS = 'crossed_cols'
EST_TYP = 'estimator_type'
HIDDEN = 'hidden_units'
TRAIN_STEPS = 'training_steps'

DEF_IND_COLS = [AGE, OCC, GENDER, STAY, CITY, MS, PROD1]
DEF_EMB_COLS = '{"Product_ID":4, "User_ID":4}'
DEF_CROSS_COLS = []
DEF_EST_TYP = DNN_REG
DEF_HIDDEN = [1000, 800, 600, 400]
DEF_TRAIN_STEPS = 10000

# predictor
PREDICTOR_DESCRIPTION = ''


def trainer_interface():
    parser = argparse.ArgumentParser(description=TRAINER_DESCRIPTION)

    parser.add_argument('--model-name', dest=MDL_NAME, type=str, default=DEF_MDL_NAME, help='model name')

    parser.add_argument('--indicators', dest=IND_COLS, type=str, nargs='+', default= DEF_IND_COLS, help='a list of features parsed as indicator columns')

    parser.add_argument('--crossings', dest=CROSS_COLS, type=str, nargs='+', default= DEF_CROSS_COLS, help='''
        Create new features by crossing existing ones.
        Format:
        [column1, column2, type, hash buckets size, dim (if emb)]
        Examples:
        --crossings '["Age", "Gender", "ind", 1]'
        --crossings '["Age", "Gender", "emb", 1, 5]'
        ''')

    parser.add_argument('--embeddings', dest=EMB_COLS, type=str, default=DEF_EMB_COLS, help='a dictionary /{"column name": dimensions/}')

    parser.add_argument('--estimator-type', dest=EST_TYP, type=str, default=DEF_EST_TYP, help='estimator type. Choose from (DNN_REG, WIDE_DEEP)')

    parser.add_argument('--hidden-units', dest=HIDDEN, type=int, nargs='+', default=DEF_HIDDEN, help='hidden units e.g. 100, 200, 100')

    parser.add_argument('--training-steps', dest=TRAIN_STEPS, type=int, default=DEF_TRAIN_STEPS, help='number of training steps')

    return parser.parse_args()

def predictor_interface():
    parser = argparse.ArgumentParser(description=PREDICTOR_DESCRIPTION)

    parser.add_argument('--model-path', dest='model_path', type=str, help='path to the model directory')

    parser.add_argument('--fpath', dest='fpath', type=str, help='path to csv file with test data')

    parser.add_argument('--records', dest='records', nargs='+', type=str, help='a dictionary {"column name": "value"}')

    parser.add_argument('--outfile', dest='outfile', type=str, help='path to  an output file to write predictions to')

    return parser.parse_args()
