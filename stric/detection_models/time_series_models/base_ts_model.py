import os
from abc import ABC, abstractmethod
from tqdm import tqdm
import time

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from stric.detection_models.time_series_models.utils import adjust_learning_rate


class BaseTSModel(nn.Module, ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def visualize(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass


class TemporalModel(nn.Module):
    def __init__(self, data, test_portion, bs_seq=100):
        super(TemporalModel, self).__init__()
        indeces = list(range(len(data)))
        self.data = data
        self.train_data = torch.utils.data.Subset(data, indeces[:-int(len(data) * test_portion)])
        self.test_data = torch.utils.data.Subset(data, indeces[-int(len(data) * test_portion):])

        self.train_loader_seq = torch.utils.data.DataLoader(self.train_data, shuffle=False, batch_size=bs_seq,
                                                            drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, shuffle=False, batch_size=bs_seq, drop_last=False)

        self.bs_seq = bs_seq

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
        criterion = torch.nn.MSELoss()
        return criterion(pred, label)

    def train_model(self, bs=100, lr=0.001, epochs=100, optimizer='Adam', show_batch_progress=False):
        self.train_loader = torch.utils.data.DataLoader(self.train_data, shuffle=True, batch_size=bs, drop_last=True)

        if show_batch_progress:
            iterator = tqdm(self.train_loader, desc="Batches ")
        else:
            iterator = self.train_loader

        optimizer = torch.optim.__dict__[optimizer](self.parameters(), lr=lr)

        self.train()
        print(f'Start Training model {self.__class__}')
        epoch_times = []
        device = list(self.parameters())[0].device
        for epoch in range(1, epochs + 1):
            adjust_learning_rate(optimizer, epoch, lr, epochs)
            start_time = time.time()
            avg_loss, counter = 0., 0
            for (x_past, _, _), (x_fut, _, _) in iterator:
                x_past = x_past.to(device).float().permute(0, 2, 1)
                x_fut = x_fut.to(device).float().permute(0, 2, 1).squeeze(-1)

                counter += 1
                self.zero_grad()
                out = self(x_past)
                loss = self.criterion(out, x_fut)
                loss.backward()

                optimizer.step()
                avg_loss += loss.item()

                if counter % 200 == 0:
                    print(f"Epoch {epoch}......Step: {counter}/{len(self.train_loader)}....... "
                          f"Average Loss for Epoch: {avg_loss / counter}")
            current_time = time.time()
            print(f"Epoch {epoch}/{epochs} Done, Total Loss: {avg_loss / len(self.train_loader):.4f}",
                  f"Total Time Elapsed: {(current_time - start_time):.4f} seconds")
            epoch_times.append(current_time - start_time)
        print(f"Total Training Time: {sum(epoch_times):.4f} seconds")

        # Needed to assess model performance (and visualize results)
        self.get_predictions()

    def get_data(self, loader):
        past, fut = [], []
        for (x_past, _, _), (x_fut, _, _) in loader:
            x_past = x_past.to('cpu').float().permute(0, 2, 1)
            x_fut = x_fut.to('cpu').float().permute(0, 2, 1).squeeze(-1)
            past.append(x_past.to('cpu'))
            fut.append(x_fut.to('cpu'))
        return torch.cat(past, 0).detach(), torch.cat(fut, 0).detach()

    def get_anomalies(self, loader):
        past, fut = [], []
        for (_, _, x_past), (_, _, x_fut) in loader:
            x_past = x_past.to('cpu').float().permute(0, 2, 1)
            x_fut = x_fut.to('cpu').float().permute(0, 2, 1).squeeze(-1)
            past.append(x_past.to('cpu'))
            fut.append(x_fut.to('cpu'))
        return torch.cat(past, 0).detach(), torch.cat(fut, 0).detach()

    def get_data_and_forward(self, loader, device):
        with torch.no_grad():
            past, fut, trend, forward, time_past, time_fut = [], [], [], [], [], []
            for (x_past, t_past, _), (x_fut, t_fut, _) in tqdm(loader, desc='Forward to get predictions '):
                x_past = x_past.to(device).float().permute(0, 2, 1)
                x_fut = x_fut.to(device).float().permute(0, 2, 1).squeeze(-1)

                trend.append(self.trend(x_past))
                forward.append(self.forward(x_past))
                past.append(x_past.to('cpu'))
                fut.append(x_fut.to('cpu'))

        past, fut = torch.cat(past, 0).detach().cpu(),  torch.cat(fut, 0).detach().cpu()
        trend, forward = torch.cat(trend, 0).detach().cpu(), torch.cat(forward, 0).detach().cpu()
        return past, fut, trend, forward

    def get_predictions(self):
        device = list(self.parameters())[0].device
        train_data, target_train, trend_tr, forward_tr, t_p_tr, t_f_tr = self.get_data_and_forward(self.train_loader_seq, device)
        test_data, target_test, trend_te, forward_te, t_p_te, t_f_te = self.get_data_and_forward(self.test_loader, device)
        # Save predictions
        self.filt_train_data, self.filt_test_data = train_data, test_data
        self.pred_train_data, self.pred_test_data = target_train, target_test
        self.trend_tr, self.trend_te = trend_tr, trend_te
        self.pred_tr, self.pred_te = forward_tr, forward_te
        self.time_labels_past_train, self.time_labels_future_train = t_p_tr, t_f_tr
        self.time_labels_past_test, self.time_labels_future_test = t_p_te, t_f_te

        # with torch.no_grad():
        #     train_data, test_data = self.get_data(self.train_loader_seq)[0], self.get_data(self.test_loader)[0]
        #     self.pred_train_data = self.get_data(self.train_loader_seq)[1]
        #     self.pred_test_data = self.get_data(self.test_loader)[1]
        #
        #     self.trend_tr = self.trend(train_data.to(device))
        #     self.trend_te = self.trend(test_data.to(device))
        #
        #     self.pred_tr = self.forward(train_data.to(device))
        #     self.pred_te = self.forward(test_data.to(device))

    def get_residuals(self, ind=0, save=False, visualize=True):
        train_residuals = self.pred_tr.detach().cpu().reshape(-1, self.pred_train_data.shape[1]) - self.pred_train_data
        test_residuals = self.pred_te.detach().cpu().reshape(-1, self.pred_train_data.shape[1]) - self.pred_test_data
        dataset_std = np.array([std for (mean, std) in self.data.dataset_statistics])

        if visualize:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes[0].hist(train_residuals.std(0), alpha=0.5, label='Train')
            axes[0].hist(test_residuals.std(0), alpha=0.5, label='Test')
            axes[0].legend()

            # axes[1].hist(train_residuals[:, ind], alpha=0.5, label='Train')
            # axes[1].hist(test_residuals[:, ind], alpha=0.5, label='Test')
            # axes[1].legend()

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

    def validate(self, loader):
        self.eval()
        avg_loss, outs = 0., []
        device = list(self.parameters())[0].device
        for (x_past, _, _), (x_fut, _, _) in loader:
            x_past = x_past.to(device).float().permute(0, 2, 1)
            x_fut = x_fut.to(device).float().permute(0, 2, 1).squeeze(-1)
            out = self(x_past)
            loss = self.criterion(out, x_fut)

            avg_loss += loss.item()
            outs.append(out.cpu())
        return avg_loss, torch.cat(outs, 0).detach()

    def visualize(self, x_lim=None, index=None, save=None):
        for ind in tqdm(range(self.train_data[0][0][0].shape[1])):
            plt.figure(figsize=(15, 5))
            tr_data, te_data = len(self.train_data), len(self.test_data)
            tr_ind, te_ind = np.array(range(tr_data)), np.array(range(tr_data, tr_data + te_data))
            tr_ind, te_ind = self.time_labels_future_train, self.time_labels_future_test

            with torch.no_grad():
                name_fig = f'{ind}-th-time-series'
                plt.title(name_fig)
                plt.plot(tr_ind, self.pred_train_data[:, ind], label='True')
                plt.plot(te_ind, self.pred_test_data[:, ind], label='Test')

                plt.plot(tr_ind, self.trend_tr[:, ind, -1].detach().cpu(), label='trend_tr')
                plt.plot(te_ind, self.trend_te[:, ind, -1].detach().cpu(), label='trend_te')

                #         plt.plot(range(tr_data),                    self.tcn_tr[:, ind, -1].detach().cpu(), label='tcn_tr')
                #         plt.plot(range(tr_data, tr_data + te_data), self.tcn_te[:, ind, -1].detach().cpu(), label='tcn_te')

                plt.plot(tr_ind, self.pred_tr[:, ind].detach().cpu(), label='pred_tr')
                plt.plot(te_ind, self.pred_te[:, ind].detach().cpu(), label='pred_te')
                plt.legend()
                if x_lim:
                    plt.xlim(x_lim)

            if save:
                save_path = os.path.join('figures', 'predictions')
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f'{name_fig}.png'))
                plt.savefig(os.path.join(save_path, f'{name_fig}.pdf'))
                plt.close()

            if index == ind:
                break
        # plt.xlim((200, 900))

    def save_predictions(self, path=''):
        # The following variables contain:
        # (trend, detrended), (periodic_part, deperiodic), (linear_part, delinear), (nonlinear, residual)
        tr_preds = self.get_components(ind=None, on_test=False, visualize=False)
        te_preds = self.get_components(ind=None, on_test=True, visualize=False)

        data = dict()
        data['train'] = {'time': None, 'data': None, 'predictions': None, 'decomposition': dict(), 'residuals': dict()}
        data['test'] = {'time': None, 'data': None, 'precitions': None, 'decomposition': dict(), 'residuals': dict()}
        data['train']['time'] = self.time_labels_future_train.numpy().tolist()  # (T_tr x 1)
        data['test']['time'] = self.time_labels_future_test.numpy().tolist()   # (T_te x 1)

        data['train']['data'] = self.pred_train_data.numpy().tolist()  # (T_tr x 1)
        data['test']['data'] = self.pred_test_data.numpy().tolist()   # (T_te x 1)

        data['train']['predictions'] = self.pred_tr.numpy().tolist()   # (T_tr x n x 1)
        data['test']['predictions'] = self.pred_te.numpy().tolist()    # (T_te x n x 1)

        data['train']['decomposition']['trend'] = tr_preds[0][0].numpy().tolist()  # (T_tr x n x 1)
        data['test']['decomposition']['trend'] = te_preds[0][0].numpy().tolist()  # (T_tr x n x 1)

        data['train']['decomposition']['seasonal'] = tr_preds[1][0].numpy().tolist()  # (T_tr x n x 1)
        data['test']['decomposition']['seasonal'] = te_preds[1][0].numpy().tolist()  # (T_tr x n x 1)

        data['train']['decomposition']['linear'] = tr_preds[2][0].numpy().tolist()  # (T_tr x n x 1)
        data['test']['decomposition']['linear'] = te_preds[2][0].numpy().tolist()  # (T_tr x n x 1)

        data['train']['residuals']['trend'] = tr_preds[0][1].numpy().tolist()  # (T_tr x n x 1)
        data['test']['residuals']['trend'] = te_preds[0][1].numpy().tolist()  # (T_tr x n x 1)

        data['train']['residuals']['seasonal'] = tr_preds[1][1].numpy().tolist()  # (T_tr x n x 1)
        data['test']['residuals']['seasonal'] = te_preds[1][1].numpy().tolist()  # (T_tr x n x 1)

        data['train']['residuals']['linear'] = tr_preds[2][1].numpy().tolist()  # (T_tr x n x 1)
        data['test']['residuals']['linear'] = te_preds[2][1].numpy().tolist()  # (T_tr x n x 1)

        data['train']['residuals']['nonlinear'] = tr_preds[3][1].numpy().tolist()  # (T_tr x n x 1)
        data['test']['residuals']['nonlinear'] = te_preds[3][1].numpy().tolist()  # (T_tr x n x 1)

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'saved_predictions.pkl'), 'wb') as file:
            pkl.dump(data, file)

    def save(self, PATH):
        torch.save({
            'model_state_dict': self.state_dict(),
        }, PATH)

    def load(self, PATH):
        checkpoint = torch.load(PATH)
        self.load_state_dict(checkpoint['model_state_dict'])