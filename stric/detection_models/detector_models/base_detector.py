import os
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Detector():
    def __init__(self, data, log_likelihood_model, kernel_type, kernel_hps, win_length, n, device, score='mean'):
        self.data, self.win_length, self.n = data, win_length, n
        self.score = score
        self.lk_model = log_likelihood_model(data,  kernel_type, kernel_hps, win_length=win_length, n=n, lam=1e-2,
                                             device=device)

    def fit(self):
        self.future_log_lik = []
        for i in tqdm(range(2 * (self.n + self.win_length - 1), len(self.data))):
            # print(f'{i}-th out of {len(data)-(2*n + win_length - 1)}', end='\r')
            curr_data = self.data[i-2*(self.n + self.win_length - 1): i]
            self.lk_model.init_data(curr_data)
            self.lk_model.fit()
            # self.div_scores.append(model.compute_divergence().item())
            g_hat_future = self.lk_model(self.lk_model.future).double()

            if self.score == 'mean':
                g_hat_future = torch.where(g_hat_future >= 0., g_hat_future, 1e-4).mean(1)
                # g_hat_future = torch.where(g_hat_future >= 0., g_hat_future, 1e-4)[:, :-self.n//2, :].sum(1)
            elif self.score == 'point':
                g_hat_future = torch.where(g_hat_future >= 0., g_hat_future, 1e-4)[:, 0, :]
            else:
                NotImplementedError

            self.future_log_lik.append(torch.log(g_hat_future).cpu())
        self.future_log_lik = torch.cat(self.future_log_lik, 1).T

    def get_future_log_lik(self):
        return torch.cat(
                             [
                                 torch.zeros(self.n + self.win_length - 1, self.future_log_lik.shape[1]),
                                 self.future_log_lik,
                                 torch.zeros(self.n + self.win_length - 1, self.future_log_lik.shape[1])
                             ], 0
                         )

    def get_anomaly_labels(self, thresholds):
        log_lik = self.future_log_lik
        a_labels = []
        for i in range(log_lik.shape[1]):
            a_labels.append(np.where(np.abs(log_lik[:, i] - log_lik[:, i].mean())> thresholds[i], 1, 0).reshape(-1, 1))
        return np.concatenate(
                                [
                                    np.zeros((self.n + self.win_length - 1, self.future_log_lik.shape[1])),
                                    np.concatenate(a_labels, 1),
                                    np.zeros((self.n + self.win_length - 1, self.future_log_lik.shape[1]))
                                ], 0
                             )

    def visualize_anomaly_labels(self, thresholds, ind=None, save=False):
        a_labels = self.get_anomaly_labels(thresholds)
        for i in range(self.data.shape[1]):
            name_fig = f'{i}-th-time-series'
            plt.figure(figsize=(15, 5))
            if a_labels.shape[1] == 1:
                plt.plot(a_labels[:, 0], label='Anomaly labels')
            else:
                plt.plot(a_labels[:, i], label='Anomaly labels')
            plt.plot(self.data[:, i], label='Input Data')
            plt.title(f'{i}-th time series')
            plt.legend()

            if save:
                save_path = os.path.join('figures', 'anomaly_labels')
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f'{name_fig}.png'))
                plt.savefig(os.path.join(save_path, f'{name_fig}.pdf'))
                plt.close()

            if i == ind:
                break

    def visualize_anomaly_scores(self, ind=None, save=False):
        a_scores = self.get_future_log_lik()
        for i in range(self.data.shape[1]):
            name_fig = f'{i}-th-time-series'
            plt.figure(figsize=(15, 5))
            if a_scores.shape[1] == 1:
                plt.plot(a_scores[:, 0], label='Anomaly Scores')
            else:
                plt.plot(a_scores[:, i], label='Anomaly Scores')
            plt.plot(self.data[:, i], label='Input Data')
            plt.title(f'{i}-th time series')
            plt.legend()

            if save:
                save_path = os.path.join('figures', 'anomaly_scores')
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f'{name_fig}.png'))
                plt.savefig(os.path.join(save_path, f'{name_fig}.pdf'))
                plt.close()

            if i == ind:
                break
