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


def get_sepsis_score(data, column_names, threshold=0.2, avg_values_filename="avg_values.joblib", min_max_scaler_filename="min_max_scaler.joblib", model_filename="model.joblib"):

    df = pd.DataFrame(data, columns=column_names)

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

    clf = load(model_filename) 
    scores = clf.predict_proba(test_df_preprocessed)[:, 1]
    labels = (scores > threshold).astype(int)
    # print(scores)
    # print(labels)
    return (scores, labels)

def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data, column_names

if __name__ == '__main__':
    # get input files
    tmp_input_dir = 'tmp_inputs'
    input_files = zipfile.ZipFile(sys.argv[1], 'r')
    input_files.extractall(tmp_input_dir)
    input_files.close()
    input_files = sorted(f for f in input_files.namelist() if os.path.isfile(os.path.join(tmp_input_dir, f)))

    # make temporary output directory
    tmp_output_dir = 'tmp_outputs'
    try:
        os.mkdir(tmp_output_dir)
    except FileExistsError:
        pass

    n = len(input_files)
    output_zip = zipfile.ZipFile(sys.argv[2], 'w')

    # make predictions for each input file
    for i in range(n):
        # read data
        input_file = os.path.join(tmp_input_dir, input_files[i])
        data, column_names = read_challenge_data(input_file)
        print(data)

        # make predictions
        if data.size != 0:
            (scores, labels) = get_sepsis_score(data, column_names)

        # write results
        file_name = os.path.split(input_files[i])[-1]
        output_file = os.path.join(tmp_output_dir, file_name)
        with open(output_file, 'w') as f:
            f.write('PredictedProbability|PredictedLabel\n')
            if data.size != 0:
                for (s, l) in zip(scores, labels):
                    f.write('%g|%d\n' % (s, l))
        output_zip.write(output_file)

    # perform clean-up
    output_zip.close()
    shutil.rmtree(tmp_input_dir)
    shutil.rmtree(tmp_output_dir)
