import os
import torch
from tqdm import tqdm
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from stric.detection_models.time_series_models.base_ts_model import TemporalModel


class ARIMATimeModel(TemporalModel):
    def __init__(self,  data, test_portion, bs_seq=100):
        super().__init__(data, test_portion, bs_seq)
        # Use a reduced dataset to perform model selection
        self.val_data = torch.utils.data.Subset(self.train_data, range(len(self.train_data) - 500,
                                                                       len(self.train_data)))
        self.val_loader = torch.utils.data.DataLoader(self.val_data, shuffle=False, batch_size=self.bs_seq,
                                                       drop_last=False)

    def set_test_dataset(self, test_data):
        # Function to be used after training (you can use the trained model on a different testset)
        self.test_data = test_data
        self.test_loader = torch.utils.data.DataLoader(self.test_data, shuffle=False, batch_size=self.bs_seq,
                                                       drop_last=False)
        self.get_predictions()

    def forward(self, x):
        raise NotImplementedError

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def criterion(self, pred, label):
        raise NotImplementedError

    def model_selection(self):
        data = self.get_data(self.val_loader)[0][:, 0, -1].numpy()
        model = pm.auto_arima(data, start_p=1, start_q=1,
                              test='adf',  # use adftest to find optimal 'd'
                              max_p=3, max_q=3,  # maximum p and q
                              m=0,  # frequency of series
                              # d=None,  # let model determine 'd'
                              d = 0,
                              seasonal=False,  # No Seasonality
                              start_P=0,
                              D=0,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

        print(model.summary())
        return model.order

    def train_model(self, bs=100, lr=0.001, epochs=100, optimizer='Adam', show_batch_progress=False):
        self.best_order = self.model_selection()
        # Needed to assess model performance (and visualize results)
        self.get_predictions()

    @staticmethod
    def rolling_prediction(self, loader, index=0):
        warnings.simplefilter('ignore', ConvergenceWarning)

        predictions = []
        for batch, _ in tqdm(loader, desc=f'Index {index}'):
            for hor in batch[0]:
                model = ARIMA(hor[:, index].numpy(), order=self.best_order)
                model_fit = model.fit()
                output = model_fit.forecast()
                predictions.append(output)
        return np.concatenate(predictions).reshape(-1, 1)

    def get_predictions(self):
        import multiprocessing
        from itertools import repeat
        self.n_processes, n_t_series = 30, self.data[0][0][0].shape[-1]
        indices_to_process = list(range(n_t_series))
        # indices_to_process = list(range(66, 95))
        a_pool = multiprocessing.Pool(self.n_processes)

        a = a_pool.starmap(self.rolling_prediction, zip(repeat(self), repeat(self.train_loader_seq), indices_to_process))
        self.pred_tr = np.concatenate(a, 1)
        b = a_pool.starmap(self.rolling_prediction, zip(repeat(self), repeat(self.test_loader), indices_to_process))
        self.pred_te = np.concatenate(b, 1)

        # self.pred_tr = self.rolling_prediction(self.train_loader_seq)
        # self.pred_te = self.rolling_prediction(self.test_loader)

    def get_residuals(self, ind=0, save=False, visualize=True):
        self.train_data = self.get_data(self.train_loader_seq)[0][:, :, -1].numpy()
        self.test_data = self.get_data(self.test_loader)[0][:, :, -1].numpy()
        train_residuals = self.pred_tr - self.train_data
        test_residuals = self.pred_te - self.test_data
        dataset_std = np.array([std for (mean, std) in self.data.dataset_statistics])

        tr_res = train_residuals.std(0).mean()
        te_res = test_residuals.std(0).mean()
        tr_res_denorm = (train_residuals.std(0) * dataset_std).mean()
        te_res_denorm = (test_residuals.std(0) * dataset_std).mean()
        print(f'Residual average std: Train {tr_res:.4f}, Test {te_res:.4f}, Gen Gap {te_res - tr_res :.4f}',
              f' Denormalized Train {tr_res_denorm:.4f}, Test {te_res_denorm:.4f}')

        if save and visualize:
            name_fig = 'test_prediction_residuals_histogram'
            save_path = os.path.join('figures', 'residuals_histogram')
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'{name_fig}.png'))
            plt.savefig(os.path.join(save_path, f'{name_fig}.pdf'))
            plt.close()

        self.results_res = {'train_res': train_residuals.std(0).mean(), 'test_res': test_residuals.std(0).mean(),
                            'train_res_denorm': tr_res_denorm, 'test_res_denorm': te_res_denorm}
        return train_residuals, test_residuals
