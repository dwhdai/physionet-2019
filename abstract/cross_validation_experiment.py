import pandas as pd
from sklearn import preprocessing, feature_selection, linear_model, model_selection, metrics, ensemble
import numpy as np
import os
import argparse
from shutil import make_archive, rmtree

# DATA = pd.read_csv("../septrain.csv", sep="\t")

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir")
args = argparser.parse_args()

data_dir = os.path.abspath(args.data_dir)

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


            

