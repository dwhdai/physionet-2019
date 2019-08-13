#!/usr/bin/env python

import xgboost as xgb

def get_sepsis_score(data, model):

    # Get number of rows in data to know which prediction to save
    row_total = data.shape[0]

    dtest = xgb.DMatrix(data)

    # score = 1 - np.exp(-l_exp_bx)
    score = model.predict(dtest)[row_total-1]
    label = score > 0.02

    return score, label

def load_sepsis_model():

    model = xgb.Booster(model_file="xgboost/xgb.model")

    return model
