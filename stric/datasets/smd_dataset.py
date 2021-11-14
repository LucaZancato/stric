import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl

from stric.datasets.base_dataset import BaseDataset


class SMDDataset(BaseDataset):
    # https://github.com/NetManAIOps/OmniAnomaly
    name = 'SMD'
    dataset_dir = 'OmniAnomaly'
    dataset_subnames = ['train', 'test']

    def load_dataset(self) -> list:
        path = os.path.join(self.dataset_path(), 'ServerMachineDataset', 'processed')
        if self.dataset_subset in self.dataset_subnames:
            listdir = os.listdir(path)
            listdir.sort()

            # Filter train or test files
            files_names = []
            for file_name in listdir:
                if self.dataset_subset in file_name:
                    files_names.append(file_name)

            # Get Test anomaly labels
            a_files_names = []
            for file_name in listdir:
                if 'label' in file_name:
                    a_files_names.append(file_name)

            print(files_names[self.dataset_index])

            # Load data: train or test
            with open(os.path.join(path, files_names[self.dataset_index]), 'rb') as file:
                data = pkl.load(file).T  # numpy array of shape T x n (number of dimensions)

            # Get anomalies (only for test set)
            if self.dataset_subset == 'train':
                a_data = np.zeros((data.shape[1],))
            else:
                with open(os.path.join(path, a_files_names[self.dataset_index]), 'rb') as file:
                    a_data = pkl.load(file)  # numpy array of shape T x n (number of dimensions)

            # Save data into proper data structure (list of DataFrames)
            dataset = []
            for d in range(data.shape[0]):
                if (data[d] == 0).sum() == len(data[d]):  # To avoid constant time series
                    data[d][0] = 1.
                pd_data = pd.DataFrame()
                pd_data['value'] = data[d].tolist()
                pd_data['is_anomaly'] = a_data.tolist()
                dataset.append(pd_data)
        else:
            raise ValueError(f'Dataset {self.name} does not contain subname {self.dataset_subset}')
        return dataset

    def preprocessing(self, dataset: list) -> list:
        if self.dataset_subset in self.dataset_subnames:
            for i in range(len(dataset)):
                if (dataset[i].isnull().any()).sum() != 0:  # There is some nan in the DataFrame
                    raise ValueError('Some values in the pandas DataFrame are none')
                dataset[i]['timestamp'] = range(len(dataset[i]))
        dataset = self.data_standardization(dataset)
        return dataset

    # similar to other dataset classes, in this case we only use 'minimum' since more data are available
    def form_dataset(self, dataset: list) -> tuple:
        X = np.concatenate([d['value'].values.reshape(-1, 1) for d in dataset], 1)
        Y = np.concatenate([d['timestamp'].values.reshape(-1, 1) for d in dataset], 1)
        Z = np.concatenate([d['is_anomaly'].values.reshape(-1, 1) for d in dataset], 1)
        return torch.tensor(X), torch.tensor(Y), torch.tensor(Z)