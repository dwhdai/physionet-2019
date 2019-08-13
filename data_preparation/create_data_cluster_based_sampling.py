# -*- coding: utf-8 -*-
# @Author: Chloe
# @Date:   2019-07-24 16:57:51
# @Last Modified by:   Chloe
# @Last Modified time: 2019-08-08 10:05:35

import argparse
import numpy as np
import pandas as pd
import os
import time
from dataset import PhysionetDatasetCNN, FEATURES, LABEL, LABS_VITALS

ratio = 0.1
num_clusters = 5
if __name__ == "__main__":
    start_time = time.time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_dir",
                           default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/")
    argparser.add_argument("--valid_dir",
                           default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/")
    argparser.add_argument("--window_size", default=24, type=int)
    argparser.add_argument("--output_dir", default=".")
    argparser.add_argument("--preprocessing_method", default="measured",
                           help="""Possible values:
                           - measured (forward-fill, add indicator variables, normalize, impute with -1),
                           - simple (only do forward-fill and impute with -1) """)
    argparser.add_argument("--kmeans_train_file",
                           default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/splits/split_0/Kmeans_clusters.csv")
    args = argparser.parse_args()
    print(args)

    window_size = args.window_size
    num_features = len(FEATURES) + len(LABS_VITALS)

    print("Loading train data")
    train_dataset = PhysionetDatasetCNN(args.train_dir)
    print("Time elapsed since start: {}".format(time.time() - start_time))
    print("Preprocessing train data")
    train_dataset.__preprocess__(method=args.preprocessing_method)
    train_dataset.__setwindow__(window_size)
    print(train_dataset.data.columns)

    print("Save preprocessed datasets")
    print("Generating train data with ratio {}".format(ratio))

    # Cluster-based sampling on train data 5x
    for r in range(5):

        data_no_outcome = train_dataset.data.iloc[train_dataset.indices_no_outcome].reset_index() # Keep index
        kmeans = pd.read_csv(args.kmeans_train_file)

        merged = data_no_outcome.merge(kmeans, on=["id", "ICULOS"])

        num_keep = int(1 / ratio) * len(train_dataset.indices_outcome)
        num_keep_per_cluster = int(num_keep / num_clusters)

        indices_keep = []
        for k in range(5):
            indices_cluster = merged[merged["cluster_label"] == k]["index"].values
            if len(indices_cluster) > 0:
                indices_keep += np.random.permutation(indices_cluster)[:num_keep_per_cluster].tolist()


        indices_train = np.random.permutation(np.concatenate((train_dataset.indices_outcome, indices_keep)))
        train_n = len(indices_train)
        print(len(indices_keep))
        print(len(indices_train))
        train_features = np.zeros((train_n, window_size, num_features))
        train_outcomes = np.zeros((train_n, 1))
        train_ids = np.zeros((train_n, 1))
        train_iculos = np.zeros((train_n, 1))
        train_filenames = np.empty((train_n), dtype="S10")

        for i in range(len(indices_train)):
            item = train_dataset.__getitem__(indices_train[i])
            train_features[i, :, :] = item[0]
            train_outcomes[i, :] = item[1]
            train_ids[i] = item[2]
            train_iculos[i] = item[3]
            train_filenames[i] = item[4]

        if ratio:
            train_filename = os.path.join(args.output_dir,
                                      "train_preprocessed_cluster_{}_window_{}_ratio_{}_{}.npz".format(args.preprocessing_method, window_size, str(ratio).replace(".", "_"), r))
        np.savez(train_filename,
                 train_features=train_features,
                 train_outcomes=train_outcomes,
                 train_ids=train_ids,
                 train_iculos=train_iculos,
                 train_filenames=train_filenames)
        print("Time elapsed since start: {}".format(time.time() - start_time))

