#!/usr/bin/env python3
import sys
import numpy as np
import os, shutil, zipfile
import pandas as pd
from sklearn import ensemble
from joblib import dump, load

VITALS_COLUMNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
LAB_COLUMNS = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets']
DEMOGRAPHIC_COLUMNS = ['Age', 'Gender']
HOSPITAL_COLUMNS = ['Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

def load_sepsis_model():
    model_filename="model.joblib"
    return load(model_filename)

def get_sepsis_score(data, model):

    threshold=0.2
    avg_values_filename="avg_values.joblib"
    min_max_scaler_filename="min_max_scaler.joblib"

    df = pd.DataFrame(data)

    # Pre-process data
    min_max_scaler = load(min_max_scaler_filename)
    avg_values = load(avg_values_filename)
    test_df_ffill = df.ffill() # Forward-fill
    test_df_imputed = test_df_ffill.fillna(avg_values) # Fill with average (from train data)
    test_df_imputed[HOSPITAL_COLUMNS] = test_df_imputed[HOSPITAL_COLUMNS].fillna(0) # Impute hospital columns with 0s
    test_df_normalized = min_max_scaler.transform(test_df_imputed[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS]) # Normalize
    test_df_preprocessed = test_df_imputed
    test_df_preprocessed[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS] = test_df_normalized
    test_df_preprocessed[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS] = test_df_preprocessed[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS].fillna(-1)


    scores = model.predict_proba(test_df_preprocessed)[:, 1]
    labels = (scores > threshold).astype(int)
    # print(scores)
    # print(labels)
    return scores, labels
