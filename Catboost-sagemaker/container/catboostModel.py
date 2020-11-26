import os
import json
import sys
import traceback
import shutil
import numpy as np
import pandas as pd
import pickle
import yaml
import pandas as pd
import numpy as np
from numpy import percentile
from src.features import preprocess,raw
from src.utils.config import PROCESSED_TRAIN_PATH,PROCESSED_TEST_PATH
from src.utils.utils import print_devider
from src.train.training import train_model,params
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.metrics import make_scorer,f1_score, confusion_matrix, roc_auc_score, auc, \
            classification_report, recall_score, precision_score

# The function to execute the training.
def train(model_dir, output_path, training_path, channel_name):
    print('\nStarting the training.')
    try:

        # Take the set of files and read them all into a single pandas dataframe
        input_files = [os.path.join(training_path, file) for file in os.listdir(training_path) ]
        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(training_path, channel_name))
        print(input_files)
        raw_data = [ pd.read_csv(file) for file in input_files ]
        data = pd.concat(raw_data)
        data.reset_index(drop=True, inplace=True)
        #preprocess the input file and save the transformed data as a pickle file
        preprocess.preprocess(dataframe=data,train=True)
        preprocess.preprocess(train=False,validation_path='./data/validation.csv')
        #deserialize the pickled file
        data = pd.read_pickle(PROCESSED_TRAIN_PATH)
        validation_data = pd.read_pickle(PROCESSED_TEST_PATH)
        
        y = data['event_type']
        X = data.drop(['event_type'], axis=1)
        
        print_devider('MLflow model')
        experiment_id, run_uuid = train_model(X, y, params, 'Access bank CTR',validation_data,model_dir)
        
        
        shutil.copytree('mlruns', f'{model_dir}/mlruns')
        print('Model saved in this path {}'.format(model_dir))
        print (" --------- ------ -------- -------- ")
        print (" --------- ------ -------- -------- ")
        print_devider(" Training completed")
        print (" --------- ------ -------- -------- ")
        
        
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)