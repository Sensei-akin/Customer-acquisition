import os
import warnings
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             roc_curve,
                             roc_auc_score,
                             precision_recall_curve,
                             average_precision_score)
from sklearn.model_selection import StratifiedKFold
# import lightgbm as lgbm
import mlflow
import catboost #import CatBoostClassifier, cv

# import sys
# sys.path.insert(0, '/home/ec2-user/SageMaker/Accessbank CTR/src')
from utils import plot_funcs as pf
from utils.utils import print_devider
from utils.config import PROCESSED_TRAIN_PATH

SEED = 0

params = {
      'model': {
          'iterations':500, 
          "depth" :6, 
          "learning_rate":0.3, 
          "l2_leaf_reg": 10, 
          "loss_function":'Logloss',
          "eval_metric":'AUC'
      },

      'fit': {
        'early_stopping_rounds': 10,
        'verbose': 10
      },

      'fold': {
        'n_splits': 5,
        'shuffle': True,
        'random_state': SEED
      }
    }

def get_scores(y_true, y_pred):
    return {
      'accuracy': accuracy_score(y_true, y_pred),
      'precision': precision_score(y_true, y_pred),
      'recall': recall_score(y_true, y_pred),
      'f1': f1_score(y_true, y_pred),
    }

def devide_by_sum(x):
    return x / x.sum()

def log_plot(args, plot_func, fp):
    if not isinstance(args, (tuple)):
        args = (args,)

    plot_func(*args, fp)
    mlflow.log_artifact(fp)
    os.remove(fp)
    print(f'Logged {fp}')

def train_model(X, y, params, exp_path,val_data,model_dir):
    fold_params = params['fold']
    model_params = params['model']
    fit_params = params['fit']

    # set mlflow experiment
    try:
        mlflow.create_experiment(exp_path)
    except (mlflow.exceptions.RestException, mlflow.exceptions.MlflowException):
        print('The specified experiment ({}) already exists.'.format(exp_path))

    mlflow.set_experiment(exp_path)

    skf = StratifiedKFold(**fold_params)
    models = []
    metrics = []

    y_proba = np.zeros(len(X))
    y_pred = np.zeros(len(X))

    feature_importances = np.zeros(X.shape[1])

    scores = defaultdict(int)

    with mlflow.start_run() as run:
        cate_features_index = np.where(X.dtypes != float)[0]

        for fold_no, (idx_train, idx_valid) in enumerate(skf.split(X, y)):
            print_devider(f'Fold: {fold_no}')

            X_train, X_valid = X.iloc[idx_train, :], X.iloc[idx_valid, :]
            y_train, y_valid = y.iloc[idx_train], y.iloc[idx_valid]

            # train model
            model = catboost.CatBoostClassifier(**model_params)
            model.fit(X_train, y_train,cat_features=cate_features_index,**fit_params,eval_set=(X_valid, y_valid))
            metrics.append({
              'name': ['AUC'],
              'values': model.evals_result_['validation']['AUC'],
              'best_iteration': model.best_iteration_
            })
            models.append(model)

            # feature importance
            feature_importances += devide_by_sum(model.feature_importances_ / skf.n_splits)

            # predict
            y_valid_proba = model.predict_proba(X_valid)[:, 1]
            y_valid_pred = model.predict(X_valid)
            y_proba[idx_valid] = y_valid_proba
            y_pred[idx_valid] = y_valid_pred

            # evaluate
            scores_test = get_scores(y_valid, y_valid_pred)

            mlflow.log_metrics({
              **scores_test,
              'best_iteration': model.best_iteration_,
            }, step=fold_no)

            print('\nScores')
            print(scores_test)
            
        print_devider('Invoking Model endpoint on validation data')
        
        valid_y = val_data['event_type']
        val_X = val_data.drop(['event_type'], axis=1)
        
    
        # predict
        y_valid_proba = model.predict_proba(val_X)[:, 1]
        y_valid_pred = model.predict(val_X)

        # evaluate
        scores_valid = get_scores(valid_y, y_valid_pred)

        # record scores
        for k, v in scores_valid.items():
            scores[k] += v 

        # log training parameters
        mlflow.log_params({
          **fold_params,
          **model_params,
          **fit_params,
          'cv': skf.__class__.__name__,
          'model': model.__class__.__name__
        })

        print_devider('Saving plots')

        # scores
        log_plot(scores, pf.scores, 'scores.png')

        # feature importance
        features = np.array(model.feature_names_)
        log_plot((features, feature_importances, 'Feature Importance'),
                 pf.feature_importance, 'feature_importance.png')

        # metric history
        log_plot(metrics, pf.metric, 'metric_history.png')

        # confusion matrix
        cm = confusion_matrix(valid_y, y_valid_pred)
        log_plot(cm, pf.confusion_matrix, 'confusion_matrix.png')

        # roc curve
        fpr, tpr, _ = roc_curve(valid_y, y_valid_pred)
        roc_auc = roc_auc_score(valid_y, y_valid_pred)
        log_plot((fpr, tpr, roc_auc), pf.roc_curve, 'roc_curve.png')

        # precision-recall curve
        pre, rec, _ = precision_recall_curve(valid_y, y_valid_proba)
        pr_auc = average_precision_score(valid_y, y_valid_pred)
        log_plot((pre, rec, pr_auc), pf.pr_curve, 'pr_curve.png')

        # After you train the model using fit(), save like this - 
        models_path = os.path.join(model_dir,'model_name')
        model.save_model(models_path) # extension not required.
        
            
        mlflow.log_artifact(models_path)
        mlflow.log_param('model_path', os.path.join(run.info.artifact_uri, models_path))
#         os.remove(models_path)

    return run.info.experiment_id, run.info.run_uuid


