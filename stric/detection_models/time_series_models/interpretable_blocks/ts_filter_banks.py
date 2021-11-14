import torch
import numpy as np
from scipy import signal


def get_single_pole(pole, n=100):
    num = [1]
    den = [1, -pole]
    t, y = signal.dimpulse((num, den, 1), n=n+1)
    return np.copy(y[0][1:][::-1])


def get_complex_pole(r, theta, n=100):
    num = [1]
    den = [1, -2 * r * np.cos(theta), r ** 2]
    t, y = signal.dimpulse((num, den, 1), n=n+2)
    return np.copy(y[0][2:][::-1])


def compute_HP_filter(T, lam=1000000):
    d1 = np.eye(T) - np.block([[np.zeros((T-1, 1)), np.eye(T-1)], [0., np.zeros((1,T-1))]]) # First difference matrix
    P = np.concatenate((np.zeros((1, T)), (d1 @ d1)[:-2], np.zeros((1, T))))
    return np.linalg.inv(np.eye(T) + lam * P.T @ P)


class RandomFilterBank(torch.nn.Module):
    def __init__(self, kernel_size=100, input_channels=1, return_all_signal=False):
        super(RandomFilterBank, self).__init__()
        self.kernel_size, self.input_channels = kernel_size, input_channels
        self.return_all_signal = return_all_signal

        self.get_impulse_responses()

        self.conv = torch.nn.Conv1d(self.input_channels, self.input_channels * self.n_features, kernel_size,
                                    padding=kernel_size,
                                    groups=self.input_channels)  # to convolve each input with its own set of filters

        self.conv.bias = torch.nn.Parameter(torch.zeros_like(self.conv.bias))

        impulse_responses = np.concatenate([imp_res.reshape(1, -1) for imp_res in self.impulse_responses], 0)
        with torch.no_grad():
            w = torch.tensor(impulse_responses).repeat(self.input_channels, 1)
            self.conv.weight[:, 0, :] = torch.nn.Parameter(w)
            self.w_0 = w

    def forward(self, x):
        if self.return_all_signal:
            return self.conv(x)[:, :, 1:-self.kernel_size].contiguous()
        else:
            return self.conv(x)[:, :, :self.kernel_size].contiguous()

    def learnable_features(self, learnable=True):
        for p in self.parameters():
            p.requires_grad = learnable

    def get_impulse_responses(self):
        raise NotImplementedError


class RFBHP(RandomFilterBank):
    def __init__(self, lams, Ts, kernel_size, input_channels, return_all_signal=False):
        self.lams, self.Ts, self.n_features = lams, Ts, len(lams)
        super(RFBHP, self).__init__(kernel_size, input_channels, return_all_signal)

    def get_impulse_responses(self):
        self.impulse_responses = []
        for lam, T in zip(self.lams, self.Ts):
            self.impulse_responses.append(compute_HP_filter(T, lam)[-1, -self.kernel_size:])


class RFBLinear(RandomFilterBank):
    def __init__(self, real_poles, complex_poles, kernel_size, input_channels, return_all_signal=False):
        self.real_poles, self.complex_poles = real_poles, complex_poles
        self.n_features = len(real_poles) + len(complex_poles)
        super(RFBLinear, self).__init__(kernel_size, input_channels, return_all_signal)

    def get_impulse_responses(self):
        self.impulse_responses = []
        for real_p in self.real_poles:
            imp_res = get_single_pole(real_p, self.kernel_size)
            self.impulse_responses.append(imp_res / max(imp_res) / 2)
        for r, theta in self.complex_poles:
            imp_res = get_complex_pole(r, theta, self.kernel_size)
            self.impulse_responses.append(imp_res / max(imp_res) / 2)