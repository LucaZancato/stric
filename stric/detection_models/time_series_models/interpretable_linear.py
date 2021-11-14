import torch

from stric.detection_models.time_series_models.base_ts_model import TemporalModel
from stric.detection_models.time_series_models.interpretable_blocks.trend_block import TrendBlock
from stric.detection_models.time_series_models.interpretable_blocks.seasonal_block import SeasonalBlock
from stric.detection_models.time_series_models.interpretable_blocks.linear_block import LinearBlock


class InterpretableLinear(TemporalModel):
    def __init__(self, data, test_portion, memory_length, pred_length, input_channels, output_channels,
                 linear_kernel_sizes,
                 HP_lams, HP_Ts,
                 real_poles, complex_poles,
                 purely_periodic_poles,
                 num_channels_TCN, kernel_size_TCN, dropout_TCN,
                 learnable_filters=False, bs_seq=100, random_init=False, regularization=False):
        super(InterpretableLinear, self).__init__(data, test_portion, memory_length, pred_length, input_channels,
                                                output_channels, linear_kernel_sizes,
                                                 HP_lams, HP_Ts,
                                                 real_poles, complex_poles,
                                                 purely_periodic_poles,
                                                 num_channels_TCN, kernel_size_TCN, dropout_TCN,
                                                 learnable_filters, bs_seq, random_init)
        self.input_channels, self.output_channels, self.random_init = input_channels, output_channels, random_init
        self.regularization=regularization

        self.trend = TrendBlock(HP_lams, HP_Ts,
                                 memory_length=memory_length,
                                 kernel_size=linear_kernel_sizes, input_channels=input_channels,
                                 learnable_filters=learnable_filters, return_all_signal=True,
                                 pred_length=pred_length, random_init=random_init)

        self.periodic_part = SeasonalBlock(purely_periodic_poles,
                                            memory_length=memory_length,
                                            kernel_size=linear_kernel_sizes, input_channels=input_channels,
                                            learnable_filters=learnable_filters, return_all_signal=True,
                                            pred_length=pred_length, random_init=random_init
                                            )

        self.linear_part = LinearBlock(real_poles, complex_poles,
                                        memory_length=memory_length,
                                        kernel_size=linear_kernel_sizes, input_channels=input_channels,
                                        learnable_filters=learnable_filters, return_all_signal=True,
                                        pred_length=pred_length, random_init=random_init
                                        )

    def forward(self, x):
        trend_past, trend_pred = self.trend(x)
        x = x - trend_past
        seas_past, seas_pred = self.periodic_part(x)
        x = x - seas_past
        linear_past, linear_pred = self.linear_part(x)
        # x = x - linear_past
        return (trend_pred + seas_pred + linear_pred).squeeze(-1)

    def criterion(self, pred, label):
        criterion = torch.nn.MSELoss()
        reg_trend = torch.abs(self.trend.linear_filter.weight.sum(1) -1).mean()
        reg_periodic = torch.abs(self.periodic_part.linear_filter.weight).mean()
        reg_linear = torch.abs(self.linear_part.linear_filter.weight).mean()
        if self.regularization:
            device = list(self.parameters())[0].device
            reg_trend += (self.trend.features.conv.weight[:, 0, :] - self.trend.features.w_0.to(device)).pow(2).mean()
            reg_periodic += (self.periodic_part.features.conv.weight[:, 0, :] - self.periodic_part.features.w_0.to(device)).pow(2).mean()
            reg_linear += (self.linear_part.features.conv.weight[:, 0, :] - self.linear_part.features.w_0.to(device)).pow(2).mean()
        return criterion(pred, label) + (reg_trend + reg_linear + reg_periodic)