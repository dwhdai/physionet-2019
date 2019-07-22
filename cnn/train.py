# -*- coding: utf-8 -*-
# @Author: Chloe
# @Date:   2019-07-22 13:06:21
# @Last Modified by:   Chloe
# @Last Modified time: 2019-07-22 13:20:04

import argparse
import torch
import pandas as pd
from model import CNN
from dataset import PhysionetDataset, PhysionetDatasetCNN, FEATURES
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

def train_model(model, loss_fn, optimizer, dataloader, num_epochs=100):

    model.train()
    for epoch in range(num_epochs):
        loss_epoch = 0.0
        for batch in dataloader:
            # Get data
            data = batch[0].float()
            target = batch[1]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            prediction = model(data)
            loss = loss_fn(prediction, target.view(-1))
            loss.backward()
            optimizer.step()

            # Get total loss
            loss_epoch += loss.item()
        print("Epoch {} -- Loss: {}".format(epoch,
                                            loss_epoch / dataloader.__len__()))

    return model

def evaluate_model(model, dataloader):

    model.eval()
    targets = []
    predictions = []
    patient_ids = []
    iculos_vals = []
    for batch in dataloader:
        data = batch[0].float()
        target = list(batch[1][:, 0].numpy())
        patient_id = list(batch[2])
        iculos = list(batch[3].numpy())
        prediction = list(torch.softmax(model(data), 1)[:, 1].detach().numpy())
        targets += target
        predictions += prediction
        patient_ids += patient_id
        iculos_vals += iculos

    results = pd.DataFrame({
        "id": patient_ids,
        "ICULOS": iculos_vals,
        "SepsisLabel": targets,
        "PredictedProbability": predictions
    })
    return results.sort_values(["id", "ICULOS"])


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_dir",
                           default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/")
    argparser.add_argument("--valid_dir",
                           default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/")
    args = argparser.parse_args()
    window_size = 8
    num_features = len(FEATURES)
    batch_size = 5
    num_epochs = 10

    train_dataset = PhysionetDatasetCNN(args.train_dir)
    train_dataset.__preprocess__()
    train_dataset.__setwindow__(window_size)
    valid_dataset = PhysionetDatasetCNN(args.valid_dir)
    valid_dataset.__preprocess__()
    valid_dataset.__setwindow__(window_size)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size, shuffle=False)

    model = CNN(input_height=window_size, input_width=num_features)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model = train_model(model, loss_fn=criterion, optimizer=optimizer, dataloader=train_dataloader, num_epochs=num_epochs)

    train_results = evaluate_model(model, train_dataloader)
    valid_results = evaluate_model(model, valid_dataloader)
    print(roc_auc_score(y_score=train_results.PredictedProbability,
                        y_true=train_results.SepsisLabel))
    print(roc_auc_score(y_score=valid_results.PredictedProbability,
                        y_true=valid_results.SepsisLabel))

