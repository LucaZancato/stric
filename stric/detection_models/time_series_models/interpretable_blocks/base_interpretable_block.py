import torch

class InterpretableBlock(torch.nn.Module):
    def __init__(self, memory_length, kernel_size, input_channels, n_filters, learnable_filters, return_all_signal=False,
                 pred_length=1, random_init=False):
        super(InterpretableBlock, self).__init__()
        self.kernel_size, self.pred_length = kernel_size, pred_length
        self.input_channels, self.n_filters, self.HP_learnable = input_channels, n_filters, learnable_filters
        self.return_all_signal, self.random_init = return_all_signal, random_init

        self.features = None
        self.linear_filter = torch.nn.Conv1d(in_channels=input_channels * self.n_filters,
                                             out_channels=input_channels,
                                             kernel_size=1, groups=input_channels)
        predictor = torch.cat([torch.normal(0., 1., (1, memory_length, pred_length))for _ in range(input_channels)], 0)

        # TO be deleted
        # predictor = torch.cat([torch.normal(0., 1., (1, 1, pred_length)) for _ in range(input_channels)], 0)
        predictor = torch.cat([torch.ones((1, 1, pred_length)) for _ in range(input_channels)], 0)
        self.linear_predictor = torch.nn.Parameter(predictor.unsqueeze(0), requires_grad=True)
        # TO be deleted
        # self.linear_predictor = torch.nn.Parameter(predictor.unsqueeze(0), requires_grad=True)

    def forward(self, x):
        features = self.features(x)   # Filters for each channel: shape (N x n_features x T_past)
        past_filtered = self.linear_filter(features)     # Filtered signal: shape (N x in_channels x T_past)

        pred_input = past_filtered.unsqueeze(-1).permute(0, 1, 3, 2)
        # Performs indepndent matrix multiplications in the number of input time series
        # future_predicted = (pred_input @ self.linear_predictor).permute(0, 1, 3, 2).squeeze(-1)

        # TO be deleted
        # pred_input = pred_input[:,:,:,-1].unsqueeze(-1)
        pred_input = pred_input[:, :, :, -1:]
        future_predicted = (pred_input @ self.linear_predictor).permute(0, 1, 3, 2).squeeze(-1)
        # TO be deleted
        return past_filtered, future_predicted