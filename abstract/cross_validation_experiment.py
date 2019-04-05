import pandas as pd
from sklearn import preprocessing, feature_selection, linear_model, model_selection, metrics, ensemble
import numpy as np
import os
import argparse
from shutil import make_archive, rmtree
from utils import *

# DATA = pd.read_csv("../septrain.csv", sep="\t")

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir")
args = argparser.parse_args()

data_dir = os.path.abspath(args.data_dir)

for i in range(10):
    train_file = os.path.join(data_dir, "split_%d/train_split_%d.csv" % (i, i))
    test_file = os.path.join(data_dir, "split_%d/test_split_%d.csv" % (i, i))

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    preprocessed_train_data, preprocessed_test_data = preprocess_data(train_data, test_data)

    X_train = preprocessed_train_data[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS].values
    X_test = preprocessed_test_data[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS].values
    y_train = preprocessed_train_data[LABEL_COLUMN].values
    y_test = preprocessed_test_data[LABEL_COLUMN].values
    
    model = ensemble.GradientBoostingClassifier(subsample=1.0, n_estimators=50, max_features="log2", max_depth=5, loss="exponential")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(metrics.classification_report(y_test, predictions))
    if model.classes_[0] == 0:
        prediction_probabilities = model.predict_proba(X_test)[:, 1]
    else:
        prediction_probabilities = model.predict_proba(X_test)[:, 0]

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


            

