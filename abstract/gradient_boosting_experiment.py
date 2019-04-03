import pandas as pd
import numpy as np
import argparse

from sklearn import preprocessing, feature_selection, linear_model, model_selection, metrics, ensemble
from cross_validation_experiment import preprocess_data

argparser = argparse.ArgumentParser()
argparser.add_argument("--train_file")
argparser.add_argument("--test_file")
args = argparser.parse_args()

TRAIN_DATA = pd.read_csv(os.path.abspath(args.train_file))
TEST_DATA = pd.read_csv(os.path.abspath(args.test_file))
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

preprocessed_train_data, preprocessed_test_data = preprocess_data(TRAIN_DATA, TEST_DATA)
X_train = preprocessed_train_data[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS].values
X_test = preprocessed_test_data[VITALS_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS].values
y_train = preprocessed_train_data[LABEL_COLUMN].values
y_test = preprocessed_test_data[LABEL_COLUMN].values


clf = ensemble.GradientBoostingClassifier()
param_dist = {"n_estimators": [50, 100, 200, 250],
              "subsample": [0.25, 0.5, 0.75, 1.0],
              "max_features": ["sqrt", "log2"],
              "loss": ["deviance", "exponential"],
              "max_depth": [3, 5, 7, 11]}
# run randomized search
n_iter_search = 20
random_search = model_selection.RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)
random_search.fit(X_train, y_train)


print(random_search.best_params_)
print(random_search.best_score_)
print(metrics.classification_report(y_true=y_train, y_pred=random_search.best_estimator_.predict(X_train)))
print(metrics.classification_report(y_true=y_test, y_pred=random_search.best_estimator_.predict(X_test)))


probs = random_search.best_estimator_.predict_proba(X_test)[:, 1]
fpr, tpr, thr = metrics.roc_curve(y_score=probs, y_true=y_test)
df = pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "thr": thr
})
df.to_csv("roc_values.csv")
