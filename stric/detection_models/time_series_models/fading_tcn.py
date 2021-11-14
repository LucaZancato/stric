import torch

from stric.detection_models.time_series_models.plain_tcn import PlainTCN
from stric.detection_models.time_series_models.tcn_block.tcn_block_fading import TCNBlockFading


class TCN_Fading(PlainTCN):
    def __init__(self, data, test_portion, memory_length, pred_length, input_channels, output_channels,
                 linear_kernel_sizes,
                 HP_lams, HP_Ts,
                 real_poles, complex_poles,
                 purely_periodic_poles,
                 num_channels_TCN, kernel_size_TCN, dropout_TCN,
                 learnable_filters=False, bs_seq=100, random_init=False, regularization=True, eta=1e-1):
        super(TCN_Fading, self).__init__(data, test_portion, memory_length, pred_length, input_channels,
                                                   output_channels, linear_kernel_sizes,
                                                   HP_lams, HP_Ts,
                                                   real_poles, complex_poles,
                                                   purely_periodic_poles,
                                                   num_channels_TCN, kernel_size_TCN, dropout_TCN,
                                                   learnable_filters, bs_seq, random_init)
        self.input_channels, self.output_channels, self.random_init = input_channels, output_channels, random_init
        self.regularization = regularization

        self.tcn = TCNBlockFading(num_channels_TCN, kernel_size_TCN, dropout_TCN,
                                     memory_length=memory_length, input_channels=input_channels,
                                     output_channels=output_channels, pred_length=pred_length)

        self.lam = torch.nn.parameter.Parameter(torch.tensor(0.9999).reshape(1, 1))
        self.lam.requires_grad = True
        self.eta = eta

    def get_Lambda(self):
        lam = self.lam.clamp(0, 1)
        return torch.cat([lam ** i for i in range(self.tcn.linear_predictor.weight.shape[1])])

    def criterion(self, pred, label):
        device = list(self.parameters())[0].device

        Lam = self.get_Lambda()
        norm_L_inv_B = (self.tcn.linear_predictor.weight.pow(2).T / Lam).sum()
        quad_form = torch.bmm(self.tcn.F * Lam.T, self.tcn.F.permute(0, 2, 1)) + self.eta * torch.eye(self.tcn.F.shape[1],
                                                                                           self.tcn.F.shape[1]).to(device)
        det = torch.diagonal(torch.cholesky(quad_form), offset=0, dim1=1, dim2=2)
        log_det = torch.log(det).mean()

        #### Old Criterion, with l1 removed on TCN #####
        criterion = torch.nn.MSELoss()
        reg_tcn = torch.abs(self.tcn.linear_tcn.weight).mean()
        old_criterion = criterion(pred, label)
        return old_criterion + norm_L_inv_B
        # return old_criterion + norm_L_inv_B + log_det