import pandas as pd
import numpy as np
import argparse
import os
from sklearn import preprocessing, feature_selection, linear_model, model_selection, metrics, ensemble
from joblib import dump, load
from utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument("--train_file")
argparser.add_argument("--test_file")
args = argparser.parse_args()

TRAIN_DATA = pd.read_csv(os.path.abspath(args.train_file))
# TEST_DATA = pd.read_csv(os.path.abspath(args.test_file))

preprocessed_train_data, preprocessed_test_data = preprocess_data(TRAIN_DATA, None)
X_train = preprocessed_train_data[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS + HOSPITAL_COLUMNS].values
# X_test = preprocessed_test_data[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS + HOSPITAL_COLUMNS].values
y_train = preprocessed_train_data[LABEL_COLUMN].values
# y_test = preprocessed_test_data[LABEL_COLUMN].values


clf = ensemble.GradientBoostingClassifier(subsample=1.0, n_estimators=50, max_features="log2", max_depth=5, loss="exponential")
param_dist = {"n_estimators": [50, 100, 200, 250],
              "subsample": [0.25, 0.5, 0.75, 1.0],
              "max_features": ["sqrt", "log2"],
              "loss": ["deviance", "exponential"],
              "max_depth": [3, 5, 7, 11]}
# run randomized search
# n_iter_search = 20
# random_search = model_selection.RandomizedSearchCV(clf, param_distributions=param_dist,
                                #    n_iter=n_iter_search, cv=5)
# random_search.fit(X_train, y_train)


# print(random_search.best_params_)
# print(random_search.best_score_)
clf.fit(X_train, y_train)
dump(clf, 'gradient_boosting_model.joblib') 
print(metrics.classification_report(y_true=y_train, y_pred=clf.predict(X_train)))
# print(metrics.classification_report(y_true=y_test, y_pred=clf.predict(X_test)))


# probs = random_search.best_estimator_.predict_proba(X_test)[:, 1]

# fpr, tpr, thr = metrics.roc_curve(y_score=probs, y_true=y_test)
# df = pd.DataFrame({
    # "fpr": fpr,
    # "tpr": tpr,
    # "thr": thr
# })
# df.to_csv("roc_values.csv")
