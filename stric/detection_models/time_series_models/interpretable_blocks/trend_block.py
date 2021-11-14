import torch

from stric.detection_models.time_series_models.interpretable_blocks.base_interpretable_block import InterpretableBlock
from stric.detection_models.time_series_models.interpretable_blocks.ts_filter_banks import RFBHP


class TrendBlock(InterpretableBlock):
    def __init__(self, HP_lams, HP_Ts, memory_length, kernel_size, input_channels, learnable_filters,
                 return_all_signal=False, pred_length=1, random_init=False):
        self.n_filters = len(HP_lams)
        super(TrendBlock, self).__init__(memory_length, kernel_size, input_channels, self.n_filters, learnable_filters,
                                          return_all_signal, pred_length, random_init)
        self.features = RFBHP(HP_lams, HP_Ts, kernel_size=kernel_size, input_channels=input_channels,
                               return_all_signal=return_all_signal)
        self.features.learnable_features(learnable_filters)

        if self.random_init:
            torch.nn.init.normal_(self.features.conv.weight, mean=0.5, std=0.1)
        torch.nn.init.constant_(self.linear_filter.weight, 1 / self.linear_filter.weight.shape[1])
        torch.nn.init.zeros_(self.linear_filter.bias)
        self.linear_filter.requires_grad_(False)