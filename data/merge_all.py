import os
import argparse
import random
import re
import numpy as np
import pandas as pd
from sklearn import model_selection
from shutil import copyfile, rmtree, make_archive
import zipfile

random.seed(1)

argparser = argparse.ArgumentParser()
argparser.add_argument("--train_files", nargs='+', default=["Z:/LKS-CHART/Projects/physionet_sepsis_project/data/training_setA.zip", "Z:/LKS-CHART/Projects/physionet_sepsis_project/data/training_setB.zip"])
args = argparser.parse_args()

train_files = args.train_files

data_file_dfs = []
subject_ids = []
for train_file in train_files:
    train_filename = os.path.abspath(train_file)
    archive = zipfile.ZipFile(train_filename, "r")

    for data_file in archive.namelist():
        if ".psv" not in data_file:
            continue
        else:
            data_file_df = pd.read_csv(archive.open(data_file), sep="|")
            data_file_dfs += [data_file_df]

            subject_id = re.search("([a-z][0-9]+).psv", data_file).groups()[0]
            subject_ids += [subject_id] * len(data_file_df)

groups = np.array(subject_ids)
merged_data_df = pd.concat(data_file_dfs)
merged_data_df["subject"] = groups

merged_data_df.to_csv("Z:/LKS-CHART/Projects/physionet_sepsis_project/data/combined_train_data.csv")