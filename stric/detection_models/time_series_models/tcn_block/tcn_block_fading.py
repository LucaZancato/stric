import torch
from torch import nn

from stric.detection_models.time_series_models.tcn_block.tcn_model import TemporalConvNet

class TCNBlockFading(torch.nn.Module):
    def __init__(self, num_inner_channels, kernel_size, dropout, memory_length, input_channels, output_channels,
                 pred_length=1):
        super(TCNBlockFading, self).__init__()
        self.features = TemporalConvNet(input_channels, num_inner_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear_tcn = nn.Linear(num_inner_channels[-1], output_channels)
        self.linear_predictor = torch.nn.Linear(memory_length, pred_length)

        # self.batch_norm = nn.BatchNorm1d(memory_length, affine=True)

    def forward(self, x):
        features = self.features(x)
        features = self.linear_tcn(features.permute(0, 2, 1)).permute(0, 2, 1)
        # features = self.batch_norm(features.permute(0, 2, 1)).permute(0, 2, 1)
        self.F = features
        future_predicted = self.linear_predictor(features)
        return future_predicted