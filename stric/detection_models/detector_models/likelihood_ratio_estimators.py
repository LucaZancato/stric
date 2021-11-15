import os
import torch
from GPR import kernels

from stric.detection_models.detector_models.utils import create_Hankel

class Subspace_likelihood_estimator(torch.nn.Module):
    def __init__(self, data, centers, kernel_type, kernel_hps, win_length, n, lam=1e-4, device='cpu'):
        super().__init__()
        self.centers, self.device = centers, device
        self.win_length, self.n, self.lam = win_length, n, lam
        self.kernel = kernels.__dict__[kernel_type](**kernel_hps)
        self.init_data(data)

    def init_data(self, data):
        self.data = data
        self.past, self.future = create_Hankel(self.data, self.win_length, self.n)
        self.past, self.future = torch.tensor(self.past).to(self.device), torch.tensor(self.future).to(self.device)
        # self.tr_data = self.past if centers is None else centers
        self.tr_data = torch.cat((self.past, self.future)) if self.centers is None else self.centers


class ULIF(Subspace_likelihood_estimator): # To model pp/pf
    def __init__(self, data, kernel_type, kernel_hps, win_length, n, lam=1e-4, centers=None, device='cpu'):
        super().__init__(data, centers, kernel_type, kernel_hps, win_length, n, lam, device)

    def fit(self):
        K = self.kernel(self.future, self.tr_data)
        H = K.T @ K / len(K)
        h = self.kernel(self.past, self.tr_data).sum(0).reshape(-1,1) / len(self.past)
        self.theta  = torch.pinverse(H + self.lam * torch.eye(H.shape[0], device=self.device)) @ h

    def forward(self, Y):
        return self.kernel(Y, self.tr_data) @ self.theta

    def compute_divergence(self):
        Y_hat_future = self.forward(self.future)
        # Y_hat_past = self.forward(self.past)
        # return - 1 / 2 * Y_hat_future.pow(2).mean() + 1 * Y_hat_past.mean() - 0.5
        # return  1 / 2 / self.n * Y_hat_future.pow(2).sum() - 1 / 2 / self.n * Y_hat_past.sum()
        # return 1 / 2 / self.n * Y_hat_future.pow(2).sum() - 1 / self.n * Y_hat_future.sum() + 0.5
        return 1 / 2 * Y_hat_future.pow(2).mean() - 1 * Y_hat_future.mean() + 0.5


class ULIFqp(Subspace_likelihood_estimator):
    def __init__(self, data, kernel_type, kernel_hps, win_length, n, lam=1e-4, centers=None, device='cpu'):
        super().__init__(data, centers, kernel_type, kernel_hps, win_length, n, lam, device)

    def fit(self):
        K = self.kernel(self.past, self.tr_data)
        H = K.T @ K / len(K)
        h = self.kernel(self.future, self.tr_data).sum(0).reshape(-1,1) / len(self.future)
        self.theta  = torch.pinverse(H + self.lam * torch.eye(H.shape[0], device=self.device)) @ h
        self.K, self.H, self.h = K, H, h
        asd =  torch.pinverse(H + self.lam * torch.eye(H.shape[0], device=self.device))
        self.asd = asd

    def forward(self, Y):
        self.K_pred = self.kernel(Y, self.tr_data)
        return (self.kernel(Y, self.tr_data) @ self.theta)

    def compute_divergence(self):
        Y_hat_future = self.forward(self.past)
        # Y_hat_past = self.forward(self.past)
        # return - 1 / 2 * Y_hat_future.pow(2).mean() + 1 * Y_hat_past.mean() - 0.5
        # return  1 / 2 / self.n * Y_hat_future.pow(2).sum() - 1 / 2 / self.n * Y_hat_past.sum()
        # return 1 / 2 / self.n * Y_hat_future.pow(2).sum() - 1 / self.n * Y_hat_future.sum() + 0.5
        return 1 / 2 * Y_hat_future.pow(2).mean() - 1 * Y_hat_future.mean() + 0.5


class RULIF(Subspace_likelihood_estimator):
    def __init__(self, data, kernel_type, kernel_hps, win_length, n, lam=1e-4, alpha=0.5, centers=None, device='cpu'):
        super().__init__(data, centers, kernel_type, kernel_hps, win_length, n, lam, device)
        self.alpha = alpha

    def fit(self):
        Kf = self.kernel(self.future, self.tr_data)
        Kp = self.kernel(self.past, self.tr_data)

        H = self.alpha * Kf.T @ Kf / len(Kf) + (self.alpha) * Kp.T @ Kp / len(Kp)
        h = Kp.mean(0).reshape(-1,1)
        self.theta  = torch.pinverse(H + self.lam * torch.eye(H.shape[0], device=self.device)) @ h

    def forward(self, Y):
        return self.kernel(Y, self.tr_data) @ self.theta

    def compute_divergence(self):
        Y_hat_future = self.forward(self.future)
        Y_hat_past = self.forward(self.past)
        return 0.5 * (self.alpha       * (Y_hat_future - 1).pow(2).mean() +
                      (1 - self.alpha) * (Y_hat_past   - 1).pow(2).mean())


class SymmetrizedRULIF(torch.nn.Module):
    def __init__(self, data, kernel_type, kernel_hps, win_length, n, lam=1e-4, alpha=0.5, centers=None, device='cpu'):
        super().__init__()
        self.kernel_type, self.kernel_hps, self.win_length = kernel_type, kernel_hps, win_length
        self.n, self.lam, self.alpha, self.centers, self.device = n, lam, alpha, centers, device
        self.init_data(data)

    def init_data(self, data):
        self.data = data[:2*self.n + self.win_length - 1]
        self.model_past_future = RULIF(self.data, self.kernel_type, self.kernel_hps, self.win_length, self.n, self.lam,
                                       self.alpha, self.centers, self.device)
        self.model_future_past = RULIF(self.data, self.kernel_type, self.kernel_hps, self.win_length, self.n, self.lam,
                                       self.alpha, self.centers, self.device)
        self.model_future_past.past = self.model_past_future.future
        self.model_future_past.future = self.model_past_future.past

    def fit(self):
        self.model_past_future.fit()
        self.model_future_past.fit()

    def compute_divergence(self):
        return self.model_past_future.compute_divergence() + self.model_future_past.compute_divergence()

class LRest2qpMultidim(Subspace_likelihood_estimator):
    def __init__(self, data, kernel_type, kernel_hps, win_length, n, lam=1e-4, centers=None, device='cpu'):
        super().__init__(data, centers, kernel_type, kernel_hps, win_length, n, lam, device)

    def fit(self):
        K = self.kernel(self.past, self.tr_data)
        H = K.T @ K / len(K)
        h = self.kernel(self.future, self.tr_data).sum(0).reshape(-1,1) / len(self.future)
        self.theta  = torch.pinverse(H + self.lam * torch.eye(H.shape[0], device=self.device)) @ h
        self.K, self.H, self.h = K, H, h

    def forward(self, Y):
        self.K_pred = self.kernel(Y, self.tr_data)
        return (self.kernel(Y, self.tr_data) @ self.theta).unsqueeze(0)

    def compute_divergence(self):
        Y_hat_future = self.forward(self.past)
        # Y_hat_past = self.forward(self.past)
        # return - 1 / 2 * Y_hat_future.pow(2).mean() + 1 * Y_hat_past.mean() - 0.5
        # return  1 / 2 / self.n * Y_hat_future.pow(2).sum() - 1 / 2 / self.n * Y_hat_past.sum()
        # return 1 / 2 / self.n * Y_hat_future.pow(2).sum() - 1 / self.n * Y_hat_future.sum() + 0.5
        return 1 / 2 * Y_hat_future.pow(2).mean() - 1 * Y_hat_future.mean() + 0.5


class LRest2qp(Subspace_likelihood_estimator):
    '''Same as ULIF_q_p class but now supports multi dimensional time series. Each series is considered
       independently.'''
    def __init__(self, data, kernel_type, kernel_hps, win_length, n, lam=1e-4, centers=None, device='cpu'):
        super().__init__(data, centers, kernel_type, kernel_hps, win_length, n, lam, device)

    def fit(self):
        K_p = torch.zeros(self.past.shape[1],     self.past.shape[0], self.tr_data.shape[0])
        K_f = torch.zeros(self.future.shape[1], self.future.shape[0], self.tr_data.shape[0])
        # FIXME: The identity creation can be improved, together with the for loop
        I = torch.zeros(self.future.shape[1], self.tr_data.shape[0], self.tr_data.shape[0])
        for i in range(self.past.shape[1]):
            K_p[i] = self.kernel(self.past[:, i].unsqueeze(1), self.tr_data[:, i].unsqueeze(1))
            K_f[i] = self.kernel(self.future[:, i].unsqueeze(1), self.tr_data[:, i].unsqueeze(1))
            I[i] = torch.eye(self.tr_data.shape[0], device=self.device)

        H = torch.bmm(K_p.permute(0, 2, 1), K_p) / self.past.shape[0]
        h = K_f.sum(1, keepdim=True) / self.future.shape[0]
        self.theta  = torch.bmm(torch.pinverse(H + self.lam * I), h.permute(0, 2, 1))

        self.K, self.H, self.h = K_p, H, h

    def forward(self, Y):
        K = torch.zeros(Y.shape[1], Y.shape[0], self.tr_data.shape[0])
        for i in range(self.past.shape[1]):
            K[i] = self.kernel(Y[:, i].unsqueeze(1), self.tr_data[:, i].unsqueeze(1))
        self.K_pred = K
        return torch.bmm(K, self.theta)

    def compute_divergence(self):
        Y_hat_future = self.forward(self.past)
        # Y_hat_past = self.forward(self.past)
        # return - 1 / 2 * Y_hat_future.pow(2).mean() + 1 * Y_hat_past.mean() - 0.5
        # return  1 / 2 / self.n * Y_hat_future.pow(2).sum() - 1 / 2 / self.n * Y_hat_past.sum()
        # return 1 / 2 / self.n * Y_hat_future.pow(2).sum() - 1 / self.n * Y_hat_future.sum() + 0.5
        return 1 / 2 * Y_hat_future.pow(2).mean() - 1 * Y_hat_future.mean() + 0.5

