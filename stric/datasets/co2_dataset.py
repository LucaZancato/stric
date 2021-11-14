import torch
import numpy as np
import pandas as pd

from stric.datasets.base_dataset import BaseDataset


class CO2Dataset(BaseDataset):
    # https://www.kaggle.com/txtrouble/carbon-emissions
    name = 'CO2'
    dataset_dir = 'CO2'
    dataset_subnames = ['CO2_kaggle']

    def load_dataset(self) -> list:
        path = os.path.join(self.dataset_path(), self.dataset_subset)
        if self.dataset_subset in self.dataset_subnames:
            listdir = os.listdir(path)
            files_names = [listdir[self.dataset_index]] if not self.dataset_index == 'all' else listdir

            dataset = []
            for file_name in files_names:
                dateparse = lambda x: pd.to_datetime(x, format='%Y%m', errors='coerce')
                dataset.append(pd.read_csv(os.path.join(path, file_name), parse_dates=['YYYYMM'], index_col='YYYYMM',
                                 date_parser=dateparse))

            # Convert to time series
            ts = dataset[0][pd.Series(pd.to_datetime(dataset[0].index, errors='coerce')).notnull().values]
            ts['Value'] = pd.to_numeric(ts['Value'], errors='coerce')
            ts.dropna(inplace=True)
            Energy_sources = ts.groupby('Description')

            dataset = []
            for desc, group in Energy_sources:
                new_pd = pd.DataFrame()
                new_pd['value'] = group.Value
                # new_pd['timestamp'] = group.index
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
        # Remove from the dataset the smaller time series
        del dataset[2]
        del dataset[3]
        del self.dataset_statistics[2]
        del self.dataset_statistics[3]
        X = np.concatenate([d['value'].values.reshape(-1, 1) for d in dataset], 1)
        Y = np.concatenate([d['timestamp'].values.reshape(-1, 1) for d in dataset], 1)
        Z = np.concatenate([d['is_anomaly'].values.reshape(-1, 1) for d in dataset], 1)
        return torch.tensor(X), torch.tensor(Y), torch.tensor(Z)
