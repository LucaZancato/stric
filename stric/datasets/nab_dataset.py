import os
import torch
import numpy as np
import pandas as pd

from stric.datasets.base_dataset import BaseDataset


class NABDataset(BaseDataset):
    # https://github.com/numenta/NAB/tree/master/data
    name = 'NAB'
    dataset_dir = 'NAB'
    dataset_subnames = ['realTweets', 'realTraffic', 'realKnownCause', 'realAWSCloudwatch', 'realAdExchange']
    dataset_indeces_dict = {'realKnownCause': [  'cpu_utilization_asg_misconfiguration.csv',
                                                 'ambient_temperature_system_failure.csv',
                                                 'nyc_taxi.csv',
                                                 'rogue_agent_key_updown.csv',
                                                 'rogue_agent_key_hold.csv',
                                                 'ec2_request_latency_system_failure.csv',
                                                 'machine_temperature_system_failure.csv']
    }

    def get_labels_path(self, path):
        new_path = os.path.join(*path.split(os.sep)[:-1], "labels", f"{self.labels_type}.json")
        if path[0] == os.sep:
            return os.path.join(os.sep, new_path)
        else:
            return new_path

    def load_dataset(self) -> list:
        self.labels_type = "onehot_combined_labels" # "onehot_combined_labels"  "onehot_combined_windows"
        self.zero_pad = True

        path = os.path.join(self.dataset_path(), self.dataset_subset)
        if self.dataset_subset in self.dataset_subnames:
            listdir = os.listdir(path)
            files_names = [listdir[self.dataset_index]] if not self.dataset_index == 'all' else listdir

            print(files_names)
            # Load dataset labels
            with open(self.get_labels_path(path)) as file:
                labels = json.load(file)

            dataset = []
            for file_name in files_names:
                loaded_dataset = pd.read_csv(os.path.join(path, file_name), index_col=None, header=0)
                loaded_dataset['is_anomaly'] = labels[os.path.join(self.dataset_subset, file_name)]
                dataset.append(loaded_dataset)
        else:
            raise ValueError(f'Dataset {self.name} does not contain subname {self.dataset_subset}')
        return dataset

    def preprocessing(self, dataset: list) -> list:
        if self.dataset_subset in self.dataset_subnames:
            for i in range(len(dataset)):
                if (dataset[i].isnull().any()).sum() != 0:  # There is some nan in the DataFrame
                    raise ValueError('Some values in the pandas DataFrame are none')
                dataset[i]['timestamp'] = range(len(dataset[i]))
                # dataset[i]['is_anomaly'] = np.zeros((dataset[i]['value'].shape))
        dataset = self.data_standardization(dataset)
        return dataset

    def zero_pad_dataset(self, dataset):
        lengths = [len(d) for d in dataset]
        padded_dataset = []
        for d in dataset:
            new_dataset = pd.DataFrame()
            new_dataset['timestamp'] = range(max(lengths))
            new_dataset['value'] = np.pad(d['value'].values, (0, max(lengths)-len(d)))
            new_dataset['is_anomaly'] = np.pad(d['is_anomaly'].values, (0, max(lengths) - len(d)))
            padded_dataset.append(new_dataset)
        return padded_dataset

    # similar to other dataset classes, in this case we only use 'minimum' since more data are available
    def form_dataset(self, dataset: list) -> tuple:
        if self.zero_pad:
            dataset = self.zero_pad_dataset(dataset)

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
