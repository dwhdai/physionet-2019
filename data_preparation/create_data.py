# -*- coding: utf-8 -*-
# @Author: Chloe
# @Date:   2019-07-24 16:57:51
# @Last Modified by:   Chloe
# @Last Modified time: 2019-08-13 16:52:24

import argparse
import numpy as np
import pandas as pd
import os
import time
from pickle import load, dump
from dataset import PhysionetDatasetCNN, FEATURES, LABEL, LABS_VITALS


def get_indices_random_sample(dataset, ratio):

    num_keep = int(1 / ratio) * len(dataset.indices_outcome)

    # Get indices of data with no outcome (sample randomly)
    # Shuffle ALL indices with no outcome and keep first
    # num_keep indices
    indices_no_outcome_keep = np.random\
        .permutation(dataset.indices_no_outcome)[:num_keep]

    # Combine with indices without outcome and shuffle
    indices_train = np.random\
        .permutation(np.concatenate((dataset.indices_outcome,
                                     indices_no_outcome_keep)))

    return indices_train


def get_indices_cluster(dataset, ratio, kmeans_df, num_clusters=5):

    # Query data with no outcome and then
    # reset index (for easier merging)
    data_no_outcome = dataset\
        .data.iloc[dataset.indices_no_outcome]\
        .reset_index()

    # Merge with kmeans
    merged = data_no_outcome.merge(kmeans_df, on=["id", "ICULOS"])

    num_keep = int(1 / ratio) * len(train_dataset.indices_outcome)
    num_keep_per_cluster = int(num_keep / num_clusters)

    indices_keep = []
    for k in range(num_clusters):
        indices_cluster = merged[merged["cluster_label"] == k]["index"].values
        if len(indices_cluster) > 0:
            # Keep a random subset of data within cluster k
            indices_keep += np.random\
                .permutation(indices_cluster)[:num_keep_per_cluster].tolist()

    # Combine indices together
    indices_train = np.random.\
        permutation(np.concatenate((dataset.indices_outcome,
                                    indices_keep)))

    return indices_train

num_iters = 10
ratios = [0.05, 0.1, 0.2, 0.5]
if __name__ == "__main__":
    start_time = time.time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_dir",
                           default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/")
    argparser.add_argument("--valid_dir",
                           default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/")
    argparser.add_argument("--window_size", default=24, type=int)
    argparser.add_argument("--output_dir", default=".")
    argparser.add_argument("--preprocessing_method", default="measured")
    argparser.add_argument("--kmeans_file",
                           default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/splits/split_0/Kmeans_clusters.csv")
    args = argparser.parse_args()
    print(args)

    # Load data
    window_size = args.window_size
    num_features = len(FEATURES) + len(LABS_VITALS)

    print("Loading train data")
    train_dataset = PhysionetDatasetCNN(args.train_dir)
    print("Time elapsed since start: {}".format(time.time() - start_time))
    print("Preprocessing train data")
    train_dataset.__preprocess__(method=args.preprocessing_method)
    train_dataset.__setwindow__(window_size)
    print(train_dataset.data.columns)
    print("Time elapsed since start: {}".format(time.time() - start_time))
    print("Loading valid data")
    valid_dataset = PhysionetDatasetCNN(args.valid_dir)
    print("Time elapsed since start: {}".format(time.time() - start_time))
    print("Preprocessing valid data")
    valid_dataset.__preprocess__(method=args.preprocessing_method)
    valid_dataset.__setwindow__(window_size)
    print("Time elapsed since start: {}".format(time.time() - start_time))

    print("Save preprocessed datasets")
    with open(os.path.join(args.output_dir, "train_preprocessed.pkl"), "wb") as f:
        dump(train_dataset, f)

    with open(os.path.join(args.output_dir, "valid_preprocessed.pkl"), "wb") as f:
        dump(valid_dataset, f)

    print("Load kmeans data")
    kmeans = pd.read_csv(args.kmeans_file)

    for iteration in range(num_iters):
        for ratio in ratios:

            """ RANDOM UNDERSAMPLING """
            print("Generating RANDOM train data with ratio {}".format(ratio))

            # Query randomly sampled indices
            indices_train_random = get_indices_random_sample(train_dataset,
                                                             ratio=ratio)
            train_n = len(indices_train_random)

            # Setup output files
            train_filename = "train_rand_{}_{}".format(str(ratio).replace(".",
                                                                          "_"),
                                                       iteration)
            train_filename_cnn = os.path.join(args.output_dir,
                                              train_filename + "_cnn.npz")
            train_filename_xgb = os.path.join(args.output_dir,
                                              train_filename + "_xgb.csv")

            # Setup matrices
            train_ids = np.zeros((train_n, 1))
            train_iculos = np.zeros((train_n, 1))
            train_filenames = np.empty((train_n), dtype="S10")
            train_features = np.zeros((train_n, window_size, num_features))
            train_outcomes = np.zeros((train_n, 1))

            # Populate matrices for CNN
            for i in range(len(indices_train_random)):
                item = train_dataset.__getitem__(indices_train_random[i])
                train_features[i, :, :] = item[0]
                train_outcomes[i, :] = item[1]
                train_ids[i] = item[2]
                train_iculos[i] = item[3]
                train_filenames[i] = item[4]

            # Get dataframe for XGB
            train_data_xgb = train_dataset.data.iloc[indices_train_random]

            # Save random sampled data
            np.savez(train_filename_cnn,
                     train_features=train_features,
                     train_outcomes=train_outcomes,
                     train_ids=train_ids,
                     train_iculos=train_iculos,
                     train_filenames=train_filenames)
            train_data_xgb.to_csv(train_filename_xgb, index=False)
            print("Time elapsed: {}".format(time.time() - start_time))



            """CLUSTER-BASED SAMPLING"""
            print("Generating CLUSTER train data with ratio {}".format(ratio))

            # Query randomly sampled indices
            indices_train_cluster = get_indices_cluster(train_dataset,
                                                        ratio=ratio,
                                                        kmeans_df=kmeans)
            train_n_cluster = len(indices_train_cluster)

            # Setup output files=
            train_filename_cluster_cnn = train_filename_cnn.replace("_rand_",
                                                                    "_cluster_")
            train_filename_cluster_xgb = train_filename_xgb.replace("_rand_",
                                                                    "_cluster_")

            # CNN setup
            train_features_cluster = np.zeros((train_n_cluster,
                                               window_size,
                                               num_features))
            train_outcomes_cluster = np.zeros((train_n_cluster, 1))
            train_ids_cluster = np.zeros((train_n_cluster, 1))
            train_iculos_cluster = np.zeros((train_n_cluster, 1))
            train_filenames_cluster = np.empty((train_n_cluster), dtype="S10")

            # Load matrices for CNN
            for i in range(len(indices_train_cluster)):
                item = train_dataset.__getitem__(indices_train_cluster[i])
                train_features_cluster[i, :, :] = item[0]
                train_outcomes_cluster[i, :] = item[1]
                train_ids_cluster[i] = item[2]
                train_iculos_cluster[i] = item[3]
                train_filenames_cluster[i] = item[4]

            # Get dataframe for XGB
            train_data_xgb_cluster = train_dataset.data.iloc[indices_train_cluster]

            # Save random sampled data
            np.savez(train_filename_cluster_cnn,
                     train_features=train_features_cluster,
                     train_outcomes=train_outcomes_cluster,
                     train_ids=train_ids_cluster,
                     train_iculos=train_iculos_cluster,
                     train_filenames=train_filenames_cluster)
            train_data_xgb_cluster.to_csv(train_filename_cluster_xgb,
                                          index=False)
            print("Time elapsed: {}".format(time.time() - start_time))

    print("Generating valid dataset")
    valid_n = valid_dataset.__len__()
    valid_features = np.zeros((valid_n, window_size, num_features))
    valid_outcomes = np.zeros((valid_n, 1))
    valid_ids = np.zeros((valid_n, 1))
    valid_iculos = np.zeros((valid_n, 1))
    valid_filenames = np.empty((valid_n), dtype="S10")
    for i in range(valid_n):
        item = valid_dataset.__getitem__(i)
        valid_features[i, :, :] = item[0]
        valid_outcomes[i, :] = item[1]
        valid_ids[i] = item[2]
        valid_iculos[i] = item[3]
        valid_filenames[i] = item[4]

    valid_filename = os.path.join(args.output_dir,
                                  "valid_cnn.npz")
    valid_filename_xgb = os.path.join(args.output_dir,
                                      "valid_xgb.csv")
    np.savez(valid_filename,
             valid_features=valid_features,
             valid_outcomes=valid_outcomes,
             valid_ids=valid_ids,
             valid_iculos=valid_iculos,
             valid_filenames=valid_filenames)
    valid_dataset.data.to_csv(valid_filename_xgb, index=False)
    print("Time elapsed since start: {}".format(time.time() - start_time))
