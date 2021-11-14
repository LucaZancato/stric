import os
import torch
import numpy as np
import pandas as pd

from stric.datasets.base_dataset import BaseDataset


class NASADataset(BaseDataset):
    # https://github.com/NetManAIOps/OmniAnomaly
    # https://github.com/khundman/telemanom
    # https://s3-us-west-2.amazonaws.com/telemanom/data.zip
    name = 'NASA'
    dataset_dir = f'OmniAnomaly{os.sep}NASA'
    dataset_subnames = ['MSL', 'SMAP']

    def load_dataset(self) -> list:
        # We assume self.dataset_subset =
        path = os.path.join(self.dataset_path())  # MSL_train or MSL_test and same for SMAP

        import pickle as pkl
        with open(os.path.join(path, 'processed', f'{self.dataset_subset}.pkl'), 'rb') as file:
            data = pkl.load(file)

        if 'test' in self.dataset_subset:
            with open(os.path.join(path, 'processed', f'{self.dataset_subset}_label.pkl'), 'rb') as file:
                labels = pkl.load(file)

        dataset = []
        for row in data.T:
            d = pd.DataFrame()
            d['value'] = row
            if 'test' in self.dataset_subset:
                d['is_anomaly'] = labels
                d['is_anomaly'] = d['is_anomaly'].astype(int)
            else:
                d['is_anomaly'] = np.zeros_like(row)
            dataset.append(d)
        return dataset

    def preprocessing(self, dataset: list) -> list:
        for i in range(len(dataset)):
            if (dataset[i].isnull().any()).sum() != 0:  # There is some nan in the DataFrame
                raise ValueError('Some values in the pandas DataFrame are none')
            if (dataset[i].eq(0).all())['value']:
                dataset[i]['value'][0] = 1
            dataset[i]['timestamp'] = range(len(dataset[i]))
        dataset = self.data_standardization(dataset)
        return dataset

    # similar to other dataset classes, in this case we only use 'minimum' since more data are available
    def form_dataset(self, dataset: list) -> tuple:
        if self.dataset_subset in self.dataset_subnames:
            lengths = [len(d['timestamp'].values) for d in dataset]
            minimum = int(np.min(lengths))
            X, Y, Z = [], [], []
            for d in dataset:
                if len(d['timestamp'].values.reshape(-1, 1)) >= minimum:
                    X.append(d['value'].values.reshape(-1, 1)[:minimum])
                    Y.append(d['timestamp'].values.reshape(-1, 1)[:minimum])
                    Z.append(d['is_anomaly'].values.reshape(-1, 1)[:minimum])

            X = np.concatenate(X, 1)
            Y = np.concatenate(Y, 1)
            Z = np.concatenate(Z, 1)
        else:
            X = np.concatenate([d['value'].values.reshape(-1, 1) for d in dataset], 1)
            Y = np.concatenate([d['timestamp'].values.reshape(-1, 1) for d in dataset], 1)
            Z = np.concatenate([d['is_anomaly'].values.reshape(-1, 1) for d in dataset], 1)
        return torch.tensor(X), torch.tensor(Y), torch.tensor(Z)
