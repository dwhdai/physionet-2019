import pandas as pd
import numpy as np
import argparse
import os
from sklearn import preprocessing, feature_selection, linear_model, model_selection, metrics, ensemble
from utils import *
from joblib import dump, load
import random


argparser = argparse.ArgumentParser()
argparser.add_argument("--train_file")
argparser.add_argument("--undersample", action="store_true")
argparser.add_argument("--output_file", default="model_undersampling.joblib")
# argparser.add_argument("--test_file")
args = argparser.parse_args()

TRAIN_DATA = pd.read_csv(os.path.abspath(args.train_file))
# TEST_DATA = pd.read_csv(os.path.abspath(args.test_file))

preprocessed_train_data, preprocessed_test_data = preprocess_data(TRAIN_DATA, None, save_preprocessor = True)
X_train = preprocessed_train_data[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS + HOSPITAL_COLUMNS].values
# X_test = preprocessed_test_data[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS + HOSPITAL_COLUMNS].values
y_train = preprocessed_train_data[LABEL_COLUMN].values
if args.undersample:
    idx_outcome = list(np.where(y_train == 1)[0])
    idx_no_outcome = list(np.where(y_train == 0)[0])
    random.shuffle(idx_no_outcome)
    idx_to_sample = idx_outcome + idx_no_outcome[:len(idx_outcome)*2]
    random.shuffle(idx_to_sample)

    print(idx_to_sample)
    X_train = X_train[idx_to_sample]
    y_train = y_train[idx_to_sample]
# y_test = preprocessed_test_data[LABEL_COLUMN].values


# Undersample model
clf = ensemble.GradientBoostingClassifier(subsample=0.75, n_estimators=250, max_features="sqrt", max_depth=11, loss="exponential")
# clf = ensemble.GradientBoostingClassifier(subsample=1.0, n_estimators=50, max_features="log2", max_depth=5, loss="exponential")
param_dist = {"n_estimators": [50, 100, 200, 250],
              "subsample": [0.25, 0.5, 0.75, 1.0],
              "max_features": ["sqrt", "log2"],
              "loss": ["deviance", "exponential"],
              "max_depth": [3, 5, 7, 11]}
# run randomized search
n_iter_search = 20
random_search = model_selection.RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)
# random_search.fit(X_train, y_train)

clf.fit(X_train, y_train)
dump(clf, args.output_file) 
# print(random_search.best_params_)
# print(random_search.best_score_)
# print(metrics.classification_report(y_true=y_train, y_pred=clf.predict(X_train)))
# print(metrics.classification_report(y_true=y_test, y_pred=clf.predict(X_test)))


# probs = clf.predict_proba(X_test)[:, 1]

# fpr, tpr, thr = metrics.roc_curve(y_score=probs, y_true=y_test)
# df = pd.DataFrame({
    # "fpr": fpr,
    # "tpr": tpr,
    # "thr": thr
# })
# df.to_csv("roc_values.csv")

# probs = []
