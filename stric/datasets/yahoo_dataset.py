import os
import torch
import numpy as np
import pandas as pd

from stric.datasets.base_dataset import BaseDataset


class YahooDataset(BaseDataset):
    # https://yahooresearch.tumblr.com/post/114590420346/a-benchmark-dataset-for-time-series-anomaly
    # Need license from Yahoo to be downloaded
    name = 'yahoo'
    dataset_dir = 'ydata-labeled-time-series-anomalies-v1_0'
    dataset_subnames = ['A1Benchmark', 'A2Benchmark', 'A3Benchmark', 'A4Benchmark']

    def load_dataset(self):
        path = os.path.join(self.dataset_path(), self.dataset_subset)
        self.dataset_index = self.dataset_index + 1 if isinstance(self.dataset_index, int) else self.dataset_index
        if self.dataset_subset == 'A1Benchmark':
            file_name, len_subnames = lambda x: f"real_{x}.csv", 67
        elif self.dataset_subset == 'A2Benchmark':
            file_name, len_subnames = lambda x: f"synthetic_{x}.csv", 100
        elif self.dataset_subset in 'A3Benchmark':
            file_name, len_subnames = lambda x: f"A3Benchmark-TS{x}.csv", 100
        elif self.dataset_subset in 'A4Benchmark':
            file_name, len_subnames = lambda x: f"A4Benchmark-TS{x}.csv", 100
        else:
            raise ValueError(f'Dataset {self.dataset_dir} does not contain subname {self.dataset_subset}')
        files_name = [file_name(self.dataset_index)] if not self.dataset_index == 'all' else [file_name(i + 1)
                                                                                           for i in range(len_subnames)]
        dataset = []
        for file_name in files_name:
            dataset.append(pd.read_csv(os.path.join(path, file_name), index_col=None, header=0))

        return dataset

    def preprocessing(self, dataset: list) -> list:
        if self.dataset_subset is None:
            raise ValueError(f'You need to choose the Benchmark on Yahoo dataset!!!')
        if self.dataset_subset == 'A1Benchmark':
            for i in range(len(dataset)):
                dataset[i] = dataset[i].rename(columns={"timestamps": "timestamp"})
                dataset[i]['timestamp'] = (dataset[i]['timestamp'] - dataset[i]['timestamp'].iloc[0]) / 3600
        elif self.dataset_subset == 'A2Benchmark':
            for i in range(len(dataset)):
                dataset[i]['timestamp'] = (dataset[i]['timestamp'] - dataset[i]['timestamp'].iloc[0]) / 3600
        elif self.dataset_subset in ['A3Benchmark', 'A4Benchmark']:
            for i in range(len(dataset)):
                dataset[i] = dataset[i].rename(columns={"anomaly": "is_anomaly", "timestamps": "timestamp"})
                dataset[i]['timestamp'] = (dataset[i]['timestamp'] - dataset[i]['timestamp'].iloc[0]) / 3600
        else:
            raise ValueError(f'Dataset {self.name} does not contain subname {self.dataset_subset}')

        dataset = self.data_standardization(dataset)
        return dataset

    def form_dataset(self, dataset: list) -> tuple:
        if self.dataset_subset in 'A1Benchmark':
            lengths = [len(d['timestamp'].values) for d in dataset]
            median = int(np.median(lengths))
            X, Y, Z, data_statistics = [], [], [], []
            for i, d in enumerate(dataset):
                if len(d['timestamp'].values.reshape(-1, 1)) >= median:
                    X.append(d['value'].values.reshape(-1, 1)[:median])
                    Y.append(d['timestamp'].values.reshape(-1, 1)[:median])
                    Z.append(d['is_anomaly'].values.reshape(-1, 1)[:median])
                    data_statistics.append(self.dataset_statistics[i])

            self.dataset_statistics = data_statistics
            X = np.concatenate(X, 1)
            Y = np.concatenate(Y, 1)
            # Y = dataset[0]['timestamp'].values.reshape(-1, )[:median]
            Z = np.concatenate(Z, 1)
        else:
            X = np.concatenate([d['value'].values.reshape(-1, 1) for d in dataset], 1)
            Y = np.concatenate([d['timestamp'].values.reshape(-1, 1) for d in dataset], 1)
            # Assuming uniform sampling
            # Y = dataset[0]['timestamp'].values.reshape(-1,)
            Z = np.concatenate([d['is_anomaly'].values.reshape(-1, 1) for d in dataset], 1)
        return torch.tensor(X), torch.tensor(Y), torch.tensor(Z)