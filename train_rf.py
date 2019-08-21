#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from train_from_intermediate import load_data


# In[21]:


import glob
glob.glob("/data/physionet_data/splits/split_0/undersampled_data/train_*_cnn.npz")


# In[22]:


test_file = "/data/physionet_data/splits/split_0/undersampled_data/valid_cnn.npz"


# In[4]:


X_train, y_train, X_test, y_test, input_shape = load_data(train_file="/data/physionet_data/splits/split_0/undersampled_data/train_cluster_0_5_0_cnn.npz",
                                                          test_file="/data/physionet_data/splits/split_0/undersampled_data/valid_cnn.npz")


# In[11]:


train_n = X_train.shape[0]
num_hours = X_train.shape[1]
num_features = X_train.shape[2]

test_n = X_test.shape[0]


rf_data_train_x = X_train.reshape(train_n, num_hours * num_features)
rf_data_train_y = y_train.reshape(train_n,)

rf_data_test_x = X_test.reshape(test_n, num_hours * num_features)
rf_data_test_y = y_test.reshape(test_n)


# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# In[13]:


rf = RandomForestClassifier(n_estimators=100, class_weight={0:1, 1:50})
rf.fit(rf_data_train_x, rf_data_train_y)


# In[16]:


y_pred = rf.predict(rf_data_test_x)
y_score = rf.predict_proba(rf_data_test_x)[:, 1]
print(classification_report(y_true=rf_data_test_y, y_pred=y_pred))
print(roc_auc_score(y_score=y_score, y_true=rf_data_test_y))


# In[17]:


from joblib import dump, load


# In[18]:


model_output = "test.joblib"
with open(model_output, "wb") as f:
    dump(rf, f)


# In[20]:


a = load(model_output)


# In[ ]:


split = ""
output_dir = os.path.join("/data/physionet_data/splits/split_0/undersampled_data/rf_output/", split)


# In[ ]:


# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
# y_test = y_test.reshape(y_test.shape[0],)
# y_pred = model.predict(X_test)
# y_pred = y_pred.reshape(y_pred.shape[0],)
y_pred_class = []

for i in y_pred:
    if i >= threshold:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)


filenames = [str(f)[2:-1]+"v" for f in npz_file["valid_filenames"]]
results = pd.DataFrame({"filename": filenames,
                       "PredictedProbability": y_pred,
                       "PredictedLabel": y_pred_class,
                       "iculos": npz_file["valid_iculos"].reshape(npz_file["valid_iculos"].shape[0],)})

print("Saving predictions to", output_dir)


for filename_ in results.filename.unique():
    df = results[results.filename == filename_][["PredictedProbability", "PredictedLabel"]]
    df.to_csv(os.path.join(output_dir, filename_), index = False, sep = "|")

