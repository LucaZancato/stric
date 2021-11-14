import os
import torch
import numpy as np
import pandas as pd
import json

from stric.datasets.base_dataset import BaseDataset


class NonlinearBenchmark(BaseDataset):
    name = 'nonlinear_benchmark'
    dataset_dir = 'nonlinear_benchmark'
    # dataset_subnames = ['giapi3', 'giapi4', 'giapi5', 'giapi6']
    dataset_subnames = ['giapi4', 'giapi5', 'giapi6']

    def load_dataset(self) -> list:
        path = os.path.join(self.dataset_path(), self.dataset_subset)
        if self.dataset_subset in self.dataset_subnames:
            listdir = os.listdir(path)
            files_names = [listdir[self.dataset_index]] if not self.dataset_index == 'all' else listdir

            dataset = []
            for i, file_name in enumerate(files_names):
                if i > 0:
                    raise NotImplementedError('Only one seed is handled in this code')
                with open(os.path.join(path, file_name), 'r') as file:
                    data = json.load(file)
                # Need input and output data
                a = np.concatenate([np.array(data['y']), np.array(data['u'])], 1)
                new_pd = pd.DataFrame(a[:, 0], columns=['value'])
                dataset.append(new_pd)
                new_pd = pd.DataFrame(a[:, 1], columns=['value'])
                dataset.append(new_pd)
        else:
            raise ValueError(f'Dataset {self.name} does not contain subname {self.dataset_subset}')
        return dataset

    def preprocessing(self, dataset: list) -> list:
        if self.dataset_subset in self.dataset_subnames:
            for i in range(len(dataset)):
                if (dataset[i].isnull().any()).sum() != 0:  # There is some nan in the DataFrame
                    raise ValueError('Some values in the pandas DataFrame are none')
                dataset[i]['timestamp'] = range(len(dataset[i]))
                dataset[i]['is_anomaly'] = np.zeros((dataset[i]['value'].shape))
        dataset = self.data_standardization(dataset)
        return dataset

    # similar to other dataset classes, in this case we only use 'minimum' since more data are available
    def form_dataset(self, dataset: list) -> tuple:
        X = np.concatenate([d['value'].values.reshape(-1, 1) for d in dataset], 1)
        Y = np.concatenate([d['timestamp'].values.reshape(-1, 1) for d in dataset], 1)
        Z = np.concatenate([d['is_anomaly'].values.reshape(-1, 1) for d in dataset], 1)
        return torch.tensor(X), torch.tensor(Y), torch.tensor(Z)