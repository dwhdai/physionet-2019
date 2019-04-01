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
argparser.add_argument("--output_dir", default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/splits/")
argparser.add_argument("--k", default=10, type=int)
args = argparser.parse_args()

train_files = args.train_files
output_dir = os.path.abspath(args.output_dir)
k = args.k

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
k_folds = model_selection.GroupKFold(n_splits=k)
split_num = 0

for train_index, test_index in k_folds.split(merged_data_df, merged_data_df, groups=groups):
    print("Creating split %d" % split_num)

    # Train data
    train_data = merged_data_df.iloc[train_index]
    train_groups = np.unique(groups[train_index])

    # Test data
    test_data = merged_data_df.iloc[test_index]
    test_groups = np.unique(groups[test_index])

    # Write merged files
    split_directory = os.path.abspath(os.path.join(output_dir, "split_" + str(split_num)))
    if not os.path.exists(output_dir):
        split_directory = os.path.join("C:/Users/PoupromC/Projects/physionet-2019/data/splits", "split_" + str(split_num))
    if not os.path.exists(split_directory):
        os.mkdir(split_directory)
    split_merged_train_filename = os.path.abspath(os.path.join(split_directory, 
                                                               "train_split_%d.csv" % split_num))
    train_data.to_csv(split_merged_train_filename, index=False)
    split_merged_test_filename = os.path.abspath(os.path.join(split_directory, 
                                                              "test_split_%d.csv" % split_num))
    test_data.to_csv(split_merged_test_filename, index=False)
    
    # Copy individual files - train, test
    split_train_directory = os.path.abspath(os.path.join(split_directory, "train"))
    train_zipfile = os.path.abspath(os.path.join(split_directory, "train"))
    if not os.path.exists(split_train_directory):
        os.mkdir(split_train_directory)
    split_test_directory = os.path.abspath(os.path.join(split_directory, "test"))
    test_zipfile = os.path.abspath(os.path.join(split_directory, "test"))
    if not os.path.exists(split_test_directory):
        os.mkdir(split_test_directory)
    
    for train_file in train_files:
        train_filename = os.path.abspath(train_file)
        archive = zipfile.ZipFile(train_filename, "r")
        for data_file in archive.namelist():
            if ".psv" not in data_file:
                continue
            else:
                if subject_id in train_groups or subject_id in test_groups:
                    extracted_csv = archive.extract(data_file, "temp")
                    subject_id = re.search("([a-z][0-9]+).psv", data_file).groups()[0]
                    split_train_group_filename = os.path.join(split_train_directory, subject_id + ".psv")
                    split_test_group_filename = os.path.join(split_test_directory, subject_id + ".psv")
                    if subject_id in train_groups:
                        copyfile(extracted_csv, split_train_group_filename)
                    elif subject_id in test_groups:
                        copyfile(extracted_csv, split_test_group_filename)

    # Cleanup 
    make_archive(train_zipfile, "zip", split_train_directory)
    rmtree(split_train_directory)
    make_archive(test_zipfile, "zip", split_test_directory)
    rmtree(split_test_directory)
  

    split_num += 1

rmtree("temp")