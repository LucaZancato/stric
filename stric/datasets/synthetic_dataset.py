import torch
import numpy as np
import pandas as pd

from stric.datasets.base_dataset import BaseDataset

class SyntheticDataset(BaseDataset):
    name = 'synthetic'
    def __init__(self, *args, **kwargs):
        self.synthetic_data = kwargs.pop('synthetic_data', False)
        super().__init__(*args, **kwargs)

    def load_dataset(self):
        dataset = []
        for i in range(self.synthetic_data.shape[1]):
            ts = pd.DataFrame()
            ts['value'] = self.synthetic_data[:, i]
            ts['timestamp'] = range(self.synthetic_data.shape[0])
            ts['is_anomaly'] = np.zeros((ts['value'].shape))
            dataset.append(ts)
        return dataset

    def preprocessing(self, dataset: list) -> list:
        dataset = self.data_standardization(dataset)
        return dataset

    def form_dataset(self, dataset: list) -> tuple:
        X = np.concatenate([d['value'].values.reshape(-1, 1) for d in dataset], 1)
        Y = np.concatenate([d['timestamp'].values.reshape(-1, 1) for d in dataset], 1)
        Z = np.concatenate([d['is_anomaly'].values.reshape(-1, 1) for d in dataset], 1)
        return torch.tensor(X), torch.tensor(Y), torch.tensor(Z)