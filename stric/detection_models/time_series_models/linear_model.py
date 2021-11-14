import torch
from stric.detection_models.time_series_models.base_ts_model import TemporalModel
from stric.detection_models.time_series_models.interpretable_blocks.linear_block import LinearBlock


class LinearModel(TemporalModel):
    def __init__(self, data, test_portion, memory_length, pred_length, input_channels, output_channels,
                 linear_kernel_sizes,
                 real_poles, complex_poles,
                 learnable_filters=False, random_init=False):
        super(LinearModel, self).__init__(data, test_portion, memory_length)
        self.input_channels, self.output_channels, self.random_init = input_channels, output_channels, random_init

        self.linear_part = LinearBlock(real_poles, complex_poles,
                                        memory_length=memory_length,
                                        kernel_size=linear_kernel_sizes, input_channels=input_channels,
                                        learnable_filters=learnable_filters, return_all_signal=True,
                                        pred_length=pred_length, random_init=random_init
                                        )

    def forward(self, x):
        linear_past, linear_pred = self.linear_part(x)
        return linear_pred.squeeze(-1)

    def criterion(self, pred, label):
        criterion = torch.nn.MSELoss()
        reg_linear = torch.abs(self.linear_part.linear_filter.weight).mean()
        return criterion(pred, label) + reg_linear