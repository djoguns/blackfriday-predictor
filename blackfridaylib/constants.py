import pkg_resources

import tensorflow as tf

SAVEDMODEL_DIR = 'savedmodels'
TENSORBOARD = 'tensorboard'
DATA_DIR = 'data'
TRAIN_FILE = pkg_resources.resource_filename('blackfridaylib', 'data/train.csv')
EVAL_FILE = pkg_resources.resource_filename('blackfridaylib', 'data/eval.csv')

# feature names
AGE='Age'
GENDER='Gender'
CITY='City_Category'
STAY='Stay_In_Current_City_Years'
OCC='Occupation'
MS='Marital_Status'
PROD1='Product_Category_1'
PROD2='Product_Category_2'
PROD3='Product_Category_3'
USER_ID = 'User_ID'
PRODUCT_ID = 'Product_ID'

# label names
PURCHASE = 'Purchase'

# Vocab lists (VL)
AGE_VL = ['26-35', '18-25', '55+', '36-45', '46-50', '0-17', '51-55']
GENDER_VL = ['M', 'F']
CITY_VL = ['A', 'B', 'C']
STAY_VL = ['0', '1', '2', '3', '4+']

# All Features
ALL_FEATS = [USER_ID, PRODUCT_ID, GENDER, AGE, OCC, CITY, STAY, MS, PROD1, PROD2, PROD3]
LABELS = [PURCHASE]

FEAT_DEF = {
    AGE : tf.feature_column.categorical_column_with_vocabulary_list(AGE, AGE_VL),
    GENDER : tf.feature_column.categorical_column_with_vocabulary_list(GENDER,GENDER_VL),
    CITY : tf.feature_column.categorical_column_with_vocabulary_list(CITY,CITY_VL),
    STAY : tf.feature_column.categorical_column_with_vocabulary_list(STAY,STAY_VL),
    OCC: tf.feature_column.categorical_column_with_identity(OCC, 21),
    MS:  tf.feature_column.categorical_column_with_identity(MS, 2),
    PROD1: tf.feature_column.categorical_column_with_identity(PROD1, 19),
    PROD2: tf.feature_column.categorical_column_with_identity(PROD2, 19),
    PROD3: tf.feature_column.categorical_column_with_identity(PROD3, 19),
    PRODUCT_ID : tf.feature_column.categorical_column_with_hash_bucket(PRODUCT_ID, hash_bucket_size=5000),
    USER_ID: tf.feature_column.categorical_column_with_hash_bucket(USER_ID, hash_bucket_size=500)
    }

# column types
IND = 'ind'
EMB = 'emb'

# data pipeline
RECORD_DTYPES = [[""], [""], [""], [""], [0], [""], [""], [0], [0], [0], [0], [0]]

# estimators
DNN_REG = 'DNN_REG'

# Deafault training paramters
DEF_STEPS = 10000

# thresholds
MAX_LYRS = 10
MAX_TRAIN_STEPS = 50000
