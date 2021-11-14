import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from stric.detection_models.time_series_models.base_ts_model import TemporalModel
from stric.detection_models.time_series_models.interpretable_blocks.trend_block import TrendBlock
from stric.detection_models.time_series_models.interpretable_blocks.seasonal_block import SeasonalBlock
from stric.detection_models.time_series_models.interpretable_blocks.linear_block import LinearBlock
from stric.detection_models.time_series_models.tcn_block.tcn_block import TCNBlock


class InterpretableTCN(TemporalModel):
    def __init__(self, data, test_portion, memory_length, pred_length, input_channels, output_channels,
                 linear_kernel_sizes,
                 HP_lams, HP_Ts,
                 real_poles, complex_poles,
                 purely_periodic_poles,
                 num_channels_TCN, kernel_size_TCN, dropout_TCN,
                 learnable_filters=False, bs_seq=100, random_init=False):
        super(InterpretableTCN, self).__init__(data, test_portion, bs_seq)
        self.input_channels, self.output_channels, self.random_init = input_channels, output_channels, random_init

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

        self.tcn = TCNBlock(num_channels_TCN, kernel_size_TCN, dropout_TCN,
                             memory_length=memory_length, input_channels=input_channels,
                             output_channels=output_channels, pred_length=pred_length)

    def forward(self, x):
        trend_past, trend_pred = self.trend(x)
        x = x - trend_past
        seas_past, seas_pred = self.periodic_part(x)
        x = x - seas_past
        linear_past, linear_pred = self.linear_part(x)
        x = x - linear_past
        tcn = self.tcn(x)
        # return (tcn + trend_pred).squeeze(-1)
        return (tcn + trend_pred + seas_pred + linear_pred).squeeze(-1)

    def criterion(self, pred, label):
        criterion = torch.nn.MSELoss()
        reg_tcn = torch.abs(self.tcn.linear_predictor.weight).mean() + torch.abs(self.tcn.linear_tcn.weight).mean()
        # Regularize filter recombination weights
        # reg_trend = torch.abs(self.trend.linear_filter.weight).mean()
        reg_trend = torch.abs(self.trend.linear_filter.weight.sum(1) -1).mean()
        reg_periodic = torch.abs(self.periodic_part.linear_filter.weight).mean()
        reg_linear = torch.abs(self.linear_part.linear_filter.weight).mean()
        # Regularize predictor recombinaion weights
        # return criterion(pred, label) + (reg_tcn + reg_trend)
        return criterion(pred, label) + (reg_tcn + reg_trend + reg_linear + reg_periodic)

    def get_data_and_forward(self, loader, device):
        with torch.no_grad():
            past, fut, trend, forward, time_past, time_future = [], [], [], [], [], []
            for (x_past, t_past, _), (x_fut, t_fut, _) in tqdm(loader, desc='Forward to get predictions '):
                x_past = x_past.to(device).float().permute(0, 2, 1)
                x_fut = x_fut.to(device).float().permute(0, 2, 1).squeeze(-1)

                trend.append(self.trend(x_past)[0])
                forward.append(self.forward(x_past).squeeze(-1))
                past.append(x_past.to('cpu'))
                fut.append(x_fut.to('cpu'))

                time_past.append(t_past[:, -1, -1].reshape(-1, 1))
                time_future.append(t_fut[:, -1, -1].reshape(-1, 1))

        past, fut = torch.cat(past, 0).detach().cpu(),  torch.cat(fut, 0).detach().cpu()
        trend, forward = torch.cat(trend, 0).detach().cpu(), torch.cat(forward, 0).detach().cpu()
        return past, fut, trend, forward, torch.cat(time_past), torch.cat(time_future)

    def extract_components(self, past, future, device):
        trend, fut_trend = self.trend(past.to(device))
        trend, fut_trend = trend.cpu(), fut_trend.cpu()
        detrended, fut_detrended = past - trend, future - fut_trend[:, :, -1]

        periodic_part, fut_periodic_part = self.periodic_part(detrended.to(device))
        periodic_part, fut_periodic_part = periodic_part.cpu(), fut_periodic_part.cpu()
        deperiodic, fut_deperiodic = detrended - periodic_part, fut_detrended - fut_periodic_part[:, :, -1]

        linear_part, fut_linear_part = self.linear_part(deperiodic.to(device))
        linear_part, fut_linear_part = linear_part.cpu(), fut_linear_part.cpu()
        delinear, fut_delinear = deperiodic - linear_part, fut_deperiodic - fut_linear_part[:, :, -1]

        nonlinear = self.tcn(delinear.to(device))
        nonlinear = nonlinear.cpu()
        predictions = nonlinear + fut_linear_part + fut_periodic_part + fut_trend

        fut_residual = future - predictions.squeeze(-1)

        return (trend, detrended, periodic_part, deperiodic, linear_part, delinear, fut_delinear, nonlinear,
                fut_residual, predictions)

    def get_components(self, ind=0, x_lim=None, visualize=True, save=False, on_test=False):
        with torch.no_grad():
            device = list(self.parameters())[0].device
            if on_test:
                loader = self.test_loader
                past, future = self.get_data(loader)
                preds_reference = self.pred_te.cpu()
                time = self.time_labels_future_test
            else:
                loader = self.train_loader_seq
                past, future = self.get_data(loader)
                preds_reference = self.pred_tr.cpu()
                time = self.time_labels_future_train

            trend, detrended, periodic_part, deperiodic, linear_part, delinear = [], [], [], [], [], []
            fut_delinear, nonlinear, fut_residual, predictions = [], [], [], []
            for (x_past, _, _), (x_fut, _, _) in tqdm(loader, desc='Get components '):
                x_past = x_past.float().permute(0, 2, 1)
                x_fut = x_fut.float().permute(0, 2, 1).squeeze(-1)
                components = self.extract_components(x_past, x_fut, device)
                # FIXME: fix this
                trend.append(components[0]), detrended.append(components[1])
                periodic_part.append(components[2]), deperiodic.append(components[3])
                linear_part.append(components[4]), delinear.append(components[5])
                fut_delinear.append(components[6]), nonlinear.append(components[7])
                fut_residual.append(components[8]), predictions.append(components[9])

            trend = torch.cat(trend)
            detrended = torch.cat(detrended)
            periodic_part = torch.cat(periodic_part)
            deperiodic = torch.cat(deperiodic)
            linear_part = torch.cat(linear_part)
            delinear = torch.cat(delinear)
            fut_delinear = torch.cat(fut_delinear)
            nonlinear = torch.cat(nonlinear)
            fut_residual = torch.cat(fut_residual)
            predictions = torch.cat(predictions)

            if torch.abs(torch.abs(predictions.squeeze(-1) - preds_reference).mean()) > 1e-3:
                raise ValueError

        if visualize:
            for i in tqdm(range(future.shape[1])):
                name_fig = f'{i}-th-time-series'
                fig, axes = plt.subplots(4, 2, figsize=(15, 5 * 4), sharey=True, sharex=True)
                plt.suptitle(name_fig)

                axes[0, 0].plot(time, future[:, i])
                axes[0, 0].plot(time, trend[:, i, -1].detach().cpu(), label='Trend')
                axes[0, 1].plot(time, detrended[:, i, -1].detach().cpu())
                axes[0, 1].set_title('Trend Removed')
                axes[0, 0].legend()

                axes[1, 0].plot(time, detrended[:, i, -1].detach().cpu())
                axes[1, 0].plot(time, periodic_part[:, i, -1].detach().cpu(), label='Periodic part')
                axes[1, 1].plot(time, deperiodic[:, i, -1].detach().cpu())
                axes[1, 1].set_title('Periodic Removed')
                axes[1, 0].legend()

                axes[2, 0].plot(time, deperiodic[:, i, -1].detach().cpu())
                axes[2, 0].plot(time, linear_part[:, i, -1].detach().cpu(), label='Linear part')
                axes[2, 1].plot(time, delinear[:, i, -1].detach().cpu())
                axes[2, 1].set_title('Linear Removed')
                axes[2, 0].legend()

                axes[3, 0].plot(time, fut_delinear[:, i].detach().cpu())
                axes[3, 0].plot(time, nonlinear[:, i].detach().cpu(), label='Nonlinear part')
                axes[3, 1].plot(time, fut_residual[:, i].detach().cpu(), label='Prediction residual')
                axes[3, 1].set_title('Non-Linear Removed')
                axes[3, 0].legend()

                if x_lim:
                    axes[0, 0].set_xlim(x_lim)

                if save:
                    save_path = os.path.join('figures', 'decomposition')
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(os.path.join(save_path, f'{name_fig}.png'))
                    plt.savefig(os.path.join(save_path, f'{name_fig}.pdf'))
                    plt.close()

                if ind is not None and i > ind:
                    break
                    
        return (trend, detrended), (periodic_part, deperiodic), (linear_part, delinear), (nonlinear, fut_residual)
