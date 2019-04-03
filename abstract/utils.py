import pandas as pd
import numpy as np
from sklearn import preprocessing, feature_selection, linear_model, model_selection, metrics, ensemble

ID_COLUMN = "subject"
LABEL_COLUMN = "SepsisLabel"
VITALS_COLUMNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
LAB_COLUMNS = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets']
DEMOGRAPHIC_COLUMNS = ['Age', 'Gender', 'HospAdmTime', 'ICULOS']
UNIT_COLUMNS = ['Unit1', 'Unit2']

def preprocess_data(train_df, test_df):

    # Forward-fill by group
    train_gr_by_subject = train_df.groupby(ID_COLUMN)
    train_df_ffill = train_gr_by_subject.ffill()

    # Impute with average
    avg_values = {}
    for feature in VITALS_COLUMNS + LAB_COLUMNS + ["Age"]:
        avg_values[feature] = train_df_ffill[feature].mean()
    train_df_imputed = train_df_ffill.fillna(avg_values)

    # Impute unit with 0s
    train_df_imputed[UNIT_COLUMNS] = train_df_imputed[UNIT_COLUMNS].fillna(0)

    # Scale between 0 and 1
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(train_df_imputed[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS])
    train_df_normalized = min_max_scaler.transform(train_df_imputed[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS])
    train_df_preprocessed = train_df_imputed
    train_df_preprocessed[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS] = train_df_normalized

    # Fill remaining values with -1 (early labs/vitals that have not yet been measured)
    train_df_preprocessed[DEMOGRAPHIC_COLUMNS] = train_df_preprocessed[DEMOGRAPHIC_COLUMNS].fillna(-1)

    # Repeat on test
    test_gr_by_subject = test_df.groupby(ID_COLUMN)
    test_df_ffill = test_gr_by_subject.ffill()
    test_df_imputed = test_df_ffill.fillna(avg_values)
    test_df_imputed[UNIT_COLUMNS] = test_df_imputed[UNIT_COLUMNS].fillna(0)
    test_df_normalized = min_max_scaler.transform(test_df_imputed[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS])
    test_df_preprocessed = test_df_imputed
    test_df_preprocessed[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS] = test_df_normalized
    test_df_preprocessed[DEMOGRAPHIC_COLUMNS] = test_df_preprocessed[DEMOGRAPHIC_COLUMNS].fillna(-1)

    return train_df_preprocessed, test_df_preprocessed