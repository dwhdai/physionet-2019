import pandas as pd
from sklearn import preprocessing, feature_selection, linear_model, model_selection, metrics, ensemble
import numpy as np
import os
import argparse
from shutil import make_archive, rmtree

# DATA = pd.read_csv("../septrain.csv", sep="\t")
ID_COLUMN = "subject"
LABEL_COLUMN = "SepsisLabel"
VITALS_COLUMNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
LAB_COLUMNS = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets']
DEMOGRAPHIC_COLUMNS = ['Age', 'Gender', 'Unit1', 'Unit2',
       'HospAdmTime', 'ICULOS']

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir")
args = argparser.parse_args()

data_dir = os.path.abspath(args.data_dir)

def preprocess_data(train_df, test_df):

    # Forward-fill by group
    train_gr_by_subject = train_df.groupby(ID_COLUMN)
    train_df_ffill = train_gr_by_subject.ffill()

    # Impute with average
    avg_values = {}
    for feature in VITALS_COLUMNS + LAB_COLUMNS + ["Age"]:
        avg_values[feature] = train_df_ffill[feature].mean()
    train_df_imputed = train_df_ffill.fillna(avg_values)

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
    test_df_normalized = min_max_scaler.transform(test_df_imputed[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS])
    test_df_preprocessed = test_df_imputed
    test_df_preprocessed[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS] = test_df_normalized
    test_df_preprocessed[DEMOGRAPHIC_COLUMNS] = test_df_preprocessed[DEMOGRAPHIC_COLUMNS].fillna(-1)

    return train_df_preprocessed, test_df_preprocessed



for i in range(10):
    train_file = os.path.joinh(data_dir, "split_%d/train_split_%d.csv" % (i, i))
    test_file = os.path.join(data_dir, "split_%d/test_split_%d.csv" % (i, i))

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    preprocessed_train_data, preprocessed_test_data = preprocess_data(train_data, test_data)

    # X = data_preprocessed[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS].values
    # y = data_preprocessed[LABEL_COLUMN].values

    X_train = preprocessed_train_data[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS].values
    X_test = preprocessed_test_data[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS].values
    y_train = preprocessed_train_data[LABEL_COLUMN].values
    y_test = preprocessed_test_data[LABEL_COLUMN].values
    
    feature_select = feature_selection.SelectKBest(k=20)
    feature_select.fit(X_train, y_train)
    X_train_top_features = feature_select.transform(X_train)
    X_test_top_features = feature_select.transform(X_test)

    model = ensemble.RandomForestClassifier(n_estimators=150)
    model.fit(X_train_top_features, y_train)
    predictions = model.predict(X_test_top_features)
    print(metrics.classification_report(y_test, predictions))
    if model.classes_[0] == 0:
        prediction_probabilities = model.predict_proba(X_test_top_features)[:, 1]
    else:
        prediction_probabilities = model.predict_proba(X_test_top_features)[:, 0]

    test_groups = test_data[ID_COLUMN]
    output_predictions_dir = "predictions_split_%d" % i
    output_predictions_zip = "predictions_split_%d" % i
    if not os.path.exists(output_predictions_dir):
        os.mkdir(output_predictions_dir)
    for test_subject in test_groups.unique():
        subject_prediction_file = os.path.join(output_predictions_dir, "%s.psv" % test_subject)
        subject_idx = np.where(test_groups == test_subject)[0]
        subject_predictions = predictions[subject_idx]
        subject_probabilities = prediction_probabilities[subject_idx]
        with open(subject_prediction_file, "w") as f:
            f.write('PredictedProbability|PredictedLabel')
            for (pred, prob) in zip(subject_predictions, subject_probabilities):
                f.write('\n%f|%d' % (prob, pred))
    
    make_archive(output_predictions_zip, "zip", output_predictions_dir)
    rmtree(output_predictions_dir)


            

