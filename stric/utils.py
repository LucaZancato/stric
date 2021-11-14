import numpy as np


def hodrick_prescott_filter(y, lam=16000000):
    # TODO find a way to automatically choose lam (regularization parameter)
    '''Simple filter to remove trend with L-2 regularization and smoothness.'''
    d1 = np.eye(len(y)) - np.block([[np.zeros((len(y)-1, 1)), np.eye(len(y)-1)], [0., np.zeros((1,len(y)-1))]]) # First difference matrix
    P = np.concatenate((np.zeros((1, len(y))), (d1 @ d1)[:-2], np.zeros((1, len(y)))))
    smoothed_y = np.linalg.inv(np.eye(len(y)) + lam * P.T @ P) @ y.reshape(-1,1)
    return smoothed_y.reshape(y.shape)