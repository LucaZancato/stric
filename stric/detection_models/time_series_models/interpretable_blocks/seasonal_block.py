import torch

from stric.detection_models.time_series_models.interpretable_blocks.base_interpretable_block import InterpretableBlock
from stric.detection_models.time_series_models.interpretable_blocks.ts_filter_banks import RFBLinear


class SeasonalBlock(InterpretableBlock):
    def __init__(self, purely_periodic_poles, memory_length, kernel_size, input_channels, learnable_filters,
                 return_all_signal=False, pred_length=1, random_init=False):
        self.n_filters = len(purely_periodic_poles)
        super(SeasonalBlock, self).__init__(memory_length, kernel_size, input_channels, self.n_filters,
                                             learnable_filters, return_all_signal, pred_length, random_init)
        self.features = RFBLinear([], purely_periodic_poles, kernel_size=kernel_size,
                                   input_channels=input_channels, return_all_signal=return_all_signal)
        self.features.learnable_features(learnable_filters)

        if self.random_init:
            torch.nn.init.normal_(self.features.conv.weight, mean=0.5, std=0.1)