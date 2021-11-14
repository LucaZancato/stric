import torch
from torch import nn

from stric.detection_models.time_series_models.base_ts_model import TemporalModel

class ZeroHoldModel(TemporalModel):
    def __init__(self, data, test_portion, memory_length, input_size, output_size, num_channels, kernel_size, dropout,
                 HP_lams, HP_Ts, HP_learnable=False, bs_seq=100, random_init=False, pred_length=1):
        super(ZeroHoldModel, self).__init__(data, test_portion, bs_seq)
        self.pred_length = pred_length
        self.linear_tcn = nn.Linear(num_channels[-1], output_size)
        self.trend = lambda x: torch.zeros_like(x)
        self.tcn = lambda x: torch.zeros_like(x)

    def forward(self, x):
        return x[:, :, -1]