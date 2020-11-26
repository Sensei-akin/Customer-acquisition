import os

DATA_DIR = 's3://aws-athena-query-results-101063123548-eu-west-1/Unsaved/2020/10/13'
TRAIN_PATH_SMS = os.path.join(DATA_DIR, '568aa3f6-1411-4200-b967-a450570d996f.csv')
TRAIN_PATH_CLICK = os.path.join(DATA_DIR, 'dbf6df4d-c16d-48bc-b8d2-f4f674084b1e.csv')
TEST_PATH = os.path.join(DATA_DIR, 'dbf6df4d-c16d-48bc-b8d2-f4f674084b1e.csv')

PROCESSED_TRAIN_PATH = os.path.join('./data', 'train.pkl')
PROCESSED_TEST_PATH = os.path.join('./data', 'test.pkl')
