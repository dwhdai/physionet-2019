import pandas as pd
from sklearn import preprocessing, feature_selection, linear_model, model_selection, metrics, ensemble
import numpy as np
import os
import argparse
from shutil import make_archive, rmtree
from utils import *
import random

# DATA = pd.read_csv("../septrain.csv", sep="\t")

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir")
argparser.add_argument("--output_dir")
argparser.add_argument("--undersample", action="store_true")
args = argparser.parse_args()

data_dir = os.path.abspath(args.data_dir)

for i in range(10):
    train_file = os.path.join(data_dir, "split_%d/train_split_%d.csv" % (i, i))
    test_file = os.path.join(data_dir, "split_%d/test_split_%d.csv" % (i, i))

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    preprocessed_train_data, preprocessed_test_data = preprocess_data(train_data, test_data)

    X_train = preprocessed_train_data[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS + HOSPITAL_COLUMNS].values
    X_test = preprocessed_test_data[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS + HOSPITAL_COLUMNS].values
    y_train = preprocessed_train_data[LABEL_COLUMN].values
    y_test = preprocessed_test_data[LABEL_COLUMN].values
    if args.undersample:
        idx_outcome = list(np.where(y_train == 1)[0])
        idx_no_outcome = list(np.where(y_train == 0)[0])
        random.shuffle(idx_no_outcome)
        idx_to_sample = idx_outcome + idx_no_outcome[:len(idx_outcome)*2]
        random.shuffle(idx_to_sample)

        print(idx_to_sample)
        X_train = X_train[idx_to_sample]
        y_train = y_train[idx_to_sample]


    # Undersample model
    model = ensemble.GradientBoostingClassifier(subsample=0.75, n_estimators=250, max_features="sqrt", max_depth=11, loss="exponential")
    # model = ensemble.GradientBoostingClassifier(subsample=1.0, n_estimators=50, max_features="log2", max_depth=5, loss="exponential")

    # No undersampling
    # {'subsample': 1.0, 'n_estimators': 50, 'max_features': 'log2', 'max_depth': 3, 'loss': 'exponential'}
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


    for threshold in [0.5, 0.4, 0.3, 0.2]:
        predictions = (prediction_probabilities > threshold).astype(int)
        test_groups = test_data[ID_COLUMN]
        output_predictions_dir = os.path.join(args.output_dir, \
                                              "predictions_split_%d_thr_%s" % (i, str(threshold).replace('.', '_')))
        output_predictions_zip = os.path.join(args.output_dir, \
                                              "predictions_split_%d_thr_%s" % (i, str(threshold).replace('.', '_')))
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
