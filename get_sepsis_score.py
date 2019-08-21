#!/usr/bin/env python3
import sys
import numpy as np
import os, shutil, zipfile
import pandas as pd
from sklearn import ensemble

from keras.models import Model, load_model
from dataset import PhysionetDatasetCNNInfer

VITALS_COLUMNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
LAB_COLUMNS = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets']
DEMOGRAPHIC_COLUMNS = ['Age', 'Gender']
HOSPITAL_COLUMNS = ['Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

def load_sepsis_model():
    model_filename = "iter_2_ratio_1_20_cluster.h5"
    return load_model(model_filename)

def get_sepsis_score(data, model):
    window_size = 24 # TODO: Change to args.window_size?

    threshold = 0.5
    # avg_values_filename="avg_values.joblib"  # avg values from train data
    # min_max_scaler_filename="min_max_scaler.joblib"  # min/max values from train data

    # mb we should trim rly high and rly low values???

    # df = pd.DataFrame(data)
    # Assuming that columns will always be in this order hmmmm
    # the fct for loading data only returns numbers no column names so idk
    # df.columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
    data_obj = PhysionetDatasetCNNInfer(data)
    data_obj.__preprocess__(method="measured")
    data_obj.__setwindow__(window_size)
    # Pre-process data

    features = data_obj.__getitem__(data_obj.__len__() - 1)[0]

    X_test = features.reshape(1, window_size, len(data_obj.features), 1)

    # Simple evaluation?
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(y_pred.shape[0],)[0]

    if y_pred >= threshold:
        label = 1
    else:
        label = 0

    # scores = model.predict_proba(test_df_preprocessed)[:, 1]
    # labels = (scores > threshold).astype(int)
    # print(scores)
    # print(labels)

    # This is will only be called for one row at a time so driver.py
    # only expects 1 score and 1 label hmmmmmmmm
    # return scores[0], labels[0]

    return y_pred, label
