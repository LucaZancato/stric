import os
import torch
import numpy as np
import pandas as pd

from stric.datasets.base_dataset import BaseDataset


class M4Dataset(BaseDataset):
    # https://www.kaggle.com/yogesh94/m4-forecasting-competition-dataset
    name = 'm4'
    dataset_dir = 'm4'
    dataset_subnames = ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']

    def load_dataset(self) -> list:
        if self.dataset_subset in self.dataset_subnames:
            train = pd.read_csv(os.path.join(self.dataset_path(), 'Train' , f'{self.dataset_subset}-train.csv'))
            # test = pd.read_csv(os.path.join('Test', f'{subname}-test.csv'))
            train = train[train.columns[1:]] # Each row is a different time series at i-th time step
            if self.dataset_index == 'all':
                raise NotImplementedError('m4 dataset is not homogeneous across time series')
            else:
                ts = pd.DataFrame()
                ts['value'] = train.iloc[self.dataset_index].values
                ts['timestamp'] = range(len(train.iloc[self.dataset_index]))
                ts['is_anomaly'] = np.zeros((ts['value'].shape))
                return [ts]
        else:
            raise ValueError(f'Dataset {self.name} does not contain subname {self.dataset_subset}')

    def preprocessing(self, dataset: list) -> list:
        if len(dataset) > 1:
            raise NotImplementedError('m4 dataset is not homogeneous across time series')
        else:
            dataset[0] = dataset[0].dropna()

        dataset = self.data_standardization(dataset)
        return dataset

    # The following is the same as in Yahoo (might be collected an made a subclass)
    def form_dataset(self, dataset: list) -> tuple:
        if self.dataset_subset in 'all':
            lengths = [len(d['timestamp'].values) for d in dataset]
            median = int(np.median(lengths))
            X, Y, Z = [], [], []
            for d in dataset:
                if len(d['timestamp'].values.reshape(-1, 1)) >= median:
                    X.append(d['value'].values.reshape(-1, 1)[:median])
                    Y.append(d['timestamp'].values.reshape(-1, 1)[:median])
                    Z.append(d['is_anomaly'].values.reshape(-1, 1)[:median])

            X = np.concatenate(X, 1)
            Y = np.concatenate(Y, 1)
            Z = np.concatenate(Z, 1)
        else:
            X = np.concatenate([d['value'].values.reshape(-1, 1) for d in dataset], 1)
            Y = np.concatenate([d['timestamp'].values.reshape(-1, 1) for d in dataset], 1)
            Z = np.concatenate([d['is_anomaly'].values.reshape(-1, 1) for d in dataset], 1)
        return torch.tensor(X), torch.tensor(Y), torch.tensor(Z)
