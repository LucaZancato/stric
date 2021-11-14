import os
import torch
import pickle as pkl

from stric.detection_models.time_series_models.base_ts_model import TemporalModel
from stric.detection_models.time_series_models.tcn_block.tcn_block import TCNBlock


class PlainTCN(TemporalModel):
    def __init__(self, data, test_portion, memory_length, pred_length, input_channels, output_channels,
                 num_channels_TCN, kernel_size_TCN, dropout_TCN, bs_seq=100, random_init=False):
        super(PlainTCN, self).__init__(data, test_portion, bs_seq)
        self.input_channels, self.output_channels, self.random_init = input_channels, output_channels, random_init

        self.tcn = TCNBlock(num_channels_TCN, kernel_size_TCN, dropout_TCN,
                             memory_length=memory_length, input_channels=input_channels,
                             output_channels=output_channels, pred_length=pred_length)

    def forward(self, x):
        tcn = self.tcn(x)
        return tcn.squeeze(-1)

    def criterion(self, pred, label):
        criterion = torch.nn.MSELoss()
        reg_tcn = torch.abs(self.tcn.linear_predictor.weight).mean() + torch.abs(self.tcn.linear_tcn.weight).mean()
        return criterion(pred, label) + reg_tcn

    def save_predictions(self, path=''):
        data = dict()
        data['train'] = {'time': None, 'data': None, 'predictions': None, 'decomposition': dict(),
                         'residuals': dict()}
        data['test'] = {'time': None, 'data': None, 'precitions': None, 'decomposition': dict(),
                        'residuals': dict()}
        data['train']['time'] = self.time_labels_future_train.numpy().tolist()  # (T_tr x 1)
        data['test']['time'] = self.time_labels_future_test.numpy().tolist()  # (T_te x 1)

        data['train']['data'] = self.pred_train_data.numpy().tolist()  # (T_tr x 1)
        data['test']['data'] = self.pred_test_data.numpy().tolist()  # (T_te x 1)

        data['train']['predictions'] = self.pred_tr.numpy().tolist()  # (T_tr x n x 1)
        data['test']['predictions'] = self.pred_te.numpy().tolist()  # (T_te x n x 1)

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'saved_predictions.pkl'), 'wb') as file:
            pkl.dump(data, file)