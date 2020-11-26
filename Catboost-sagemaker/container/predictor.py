 # This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
import numpy as np
import os
import json
import pickle
import io
import sys
import signal
import traceback
import yaml
from src.utils.utils import print_devider
import flask
import pandas as pd
from catboost import CatBoostClassifier

from src.features.preprocess import *
from src.utils.utils import input_columns

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model/model_name')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = os.listdir('/opt/ml/model/') is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        print_devider('Invoking model endpoint')
        
        data = flask.request.data.decode('utf-8')
        data = io.StringIO(data)
        data = pd.read_csv(data, sep = '|',usecols= input_columns) 
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')
    
    msisdn = data.msisdn
    data = handle_nan(data,fillna='missing',drop_outliers=True)
    data = data.reindex(columns = input_columns)
#     data = drop_cols(data,columns=['msisdn.1'])

    print('\nInvoked with {} records'.format(data.shape[0]))

    print('\nAlmost there')
    # And then, later load - 
    model = CatBoostClassifier()      # parameters not required.
    model.load_model(model_path)
    # Do the prediction
    predictions = np.round(model.predict_proba(data)[:, 1],3)
    
    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({'msisdn':msisdn,'results':predictions.flatten()}).to_csv(out,index=False,
                                                                       sep='|',header=['msisdn','catboost_access_bank'])
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')

