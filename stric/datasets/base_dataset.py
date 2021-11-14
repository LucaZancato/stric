import os
from typing import Union

from abc import ABC, abstractmethod
from torch.utils.data.dataset import Dataset

from stric.utils import hodrick_prescott_filter


class BaseDataset(Dataset, ABC):
    '''
    The structure of the dataset should be:
        data_dir/dataset_name/dataset_subset/dataset_index
        Have a look at yahoo dataset to have an example of the folder tree.
        The daset is composed by 3 main data structures saved as a torch.Tensor (N_samples x N_time_series):
            - time series values
            - time series indeces
            - time series anomaly values
    '''
    def __init__(
            self,
            past_len: int,
            fut_len: int,
            seed: int = 0,
            data_path: str = os.path.abspath(os.path.join('..', 'data')),
            dataset_subset: str = '',
            dataset_index: Union[int, str] = 0,
            normalize: bool = True,
            detrend: bool = False,
    ):
        self.data_path = data_path
        self.dataset_subset = dataset_subset
        self.dataset_index = dataset_index
        self.normalize, self.detrend = normalize, detrend

        self.past_len, self.fut_len = past_len, fut_len

        self.seed = seed

        self.load_and_preprocess()

    def dataset_path(self):
        return os.path.join(self.data_path, self.dataset_dir)

    def load_and_preprocess(self):
        dataset = self.load_dataset()
        dataset = self.preprocessing(dataset)
        self.data, self.timestamps, self.anomaly_values = self.form_dataset(dataset)

    def data_standardization(self, dataset):
        self.dataset_statistics, self.n_timeseries = [], len(dataset)
        if self.normalize:
            for i in range(len(dataset)):
                mean, std = dataset[i]['value'].mean(), dataset[i]['value'].std()
                dataset[i]['value'] = (dataset[i]['value'] - mean) / std
                self.dataset_statistics.append((mean, std))
        if self.detrend:
            for i in range(len(dataset)):
                dataset[i]['value'] -= hodrick_prescott_filter(dataset[i]['value'].values, lam=16000000)
        if self.normalize and self.detrend:
            for i in range(len(dataset)):
                dataset[i]['value'] = (dataset[i]['value'] - dataset[i]['value'].mean()) / dataset[i]['value'].std()
        return dataset

    @abstractmethod
    def load_dataset(self) -> list:
        '''Load the dataset from the dataset path. Its output is a list to be fed into the preprocessing method'''
        pass

    @abstractmethod
    def preprocessing(self, dataset: list) -> list:
        '''Preprocess the data from the method load_dataset and returns a list to be fed to form_dataset'''
        pass

    @abstractmethod
    def form_dataset(self, dataset: list) -> tuple:
        '''Produces a tuple of length 3 containing torch.Tensors of the proper size to represent the dataset:
            - time series values
            - time series indeces
            - time series anomaly values
        '''
        pass

    def __getitem__(self, idx):
        # This is going to partition the trajectory into non overlapping segments
        past_win = (
                    self.data[idx : idx + self.past_len, :],
                    self.timestamps[idx : idx + self.past_len],
                    self.anomaly_values[idx : idx + self.past_len, :]
                    )
        fut_win = (
                    self.data[idx + self.past_len : idx + self.past_len + self.fut_len, :],
                    self.timestamps[idx + self.past_len : idx + self.past_len + self.fut_len],
                    self.anomaly_values[idx + self.past_len : idx + self.past_len + self.fut_len, :],
                   )
        return (past_win, fut_win)

    def __len__(self):
        return self.data.shape[0] - (self.past_len + self.fut_len) + 1