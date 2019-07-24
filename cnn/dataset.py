
from torch.utils.data import Dataset
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os


FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
            'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST',
            'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
            'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium',
            'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI',
            'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
            'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
LABEL = ['SepsisLabel']

# Labs and Vitals that needed indicators
LABS_VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets']


class PhysionetDataset(Dataset):

    """
    Example usage:
    datadir = "Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/"
    dataset = PhysionetDataset(datadir)
    dataset.__preprocess__()

    # Use with PyTorch
    dataloader = DataLoader(dataset, batch_size=5)
    for i, batch in enumerate(dataloader):
        row_data = batch[0]
        label = batch[1]
        # Run model on batch!!

    # Use entire dataframe
    data = dataset.data


    Attributes:
        data (TYPE): pandas.DataFrame
    """

    def __init__(self, input_directory):

        filenames = os.listdir(input_directory)

        # Read all filenames
        all_patients_dfs = []
        for filename in filenames:
            patient_id = filename.split(".")[0]
            patient_df = pd.read_csv(os.path.join(input_directory, filename),
                                     sep="|")
            patient_df["id"] = patient_id  # Add patient ID to data

            all_patients_dfs += [patient_df]

        self.data = pd.concat(all_patients_dfs)

    def __len__(self):
        return len(self.data)

    # Simple preprocessing
    def __preprocess__(self):        

        # Forward fill
        self.data = self.data.groupby("id").ffill()

        # Add indicator variables & fill with means for labs/vitals
        for feature in LABS_VITALS:
            # Add indicator variable for each labs/vitals "xxx" with name "xxx_measured" and fill with 1 (measured) or 0 (not measured)
            self.data[feature + "_measured"] = [int(not(val)) for val in self.data[feature].isna().tolist()]
            # Fill NaNs in labs/vitals into averages for each patient
            self.data[feature].fillna(self.data.groupby("id")[feature].transform("mean"), inplace = True)
        
        # Fill the rest NaNs with -1
        self.data = self.data.fillna(-1)

        # Normalization for certain columns
        selected_normalize = self.data.drop(["id", "Unit1", "Unit2", 'SepsisLabel'], axis=1)
        x = selected_normalize.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self.data[selected_normalize.columns.tolist()] = x_scaled
        

    # Returns 1 row of data
    def __getitem__(self, index):
        patient_id = self.data.iloc[index]["id"]
        iculos = self.data.iloc[index]["ICULOS"]
        return (self.data.iloc[index][FEATURES].values.astype(float),
                self.data.iloc[index][LABEL].values.astype(int),
                patient_id,
                iculos)


# Inherit from PhysionetDataset --> use same preprocessing
class PhysionetDatasetCNN(PhysionetDataset):

    """
    Example usage:
    datadir = "Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/"
    dataset = PhysionetDatasetCNN(datadir)
    dataset.__preprocess__()
    dataset.__setwindow__(window = 8) # Generates 8 hour windows!

    # Use with PyTorch
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i, batch in enumerate(dataloader):
        cnn_data = batch[0]
        label = batch[1]
        # blah blah blah cnn

    Attributes:
        window (TYPE): Description
    """

    def __setwindow__(self, window):
        self.window = window

    # Override the __getitem__ function to return data for CNN instead
    # of one row
    def __getitem__(self, index):

        updated_features = self.data.columns.tolist()
        updated_features.remove("id")
        updated_features.remove("SepsisLabel")
        patient_id = self.data.iloc[index]["id"]
        iculos = self.data.iloc[index]["ICULOS"]

        if index < self.window:
            window_data = self.data.iloc[:index + 1]
        else:
            window_data = self.data.iloc[index + 1 - self.window: index + 1]

        outcome = window_data[LABEL].values[-1]

        if (window_data["id"].nunique() == 1 and
                len(window_data) == self.window):
            data = window_data[updated_features].values
        else:
            data = np.zeros((self.window, len(updated_features)))
            clipped_window = window_data[window_data["id"] == patient_id]
            data[-len(clipped_window):, :] = clipped_window[updated_features].values

        # data has shape (self.window, len(FEATURES))
        return (data, outcome, patient_id, iculos)


