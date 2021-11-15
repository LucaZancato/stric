import torch
import numpy as np
from sklearn import svm
from stric.detection_models.time_series_models.base_ts_model import TemporalModel

class OneClassSVMTimeModel(TemporalModel):
    def __init__(self,  data, test_portion, bs_seq=100):
        super().__init__(data, test_portion, bs_seq)
        # Use a reduced dataset to perform model selection
        self.val_data = torch.utils.data.Subset(self.train_data, range(len(self.train_data) - 500,
                                                                       len(self.train_data)))
        self.val_loader = torch.utils.data.DataLoader(self.val_data, shuffle=False, batch_size=self.bs_seq,
                                                       drop_last=False)

        self.model = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

    @staticmethod
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'same') / w

    def get_anomalies_scores(self, loader):
        X = self.get_data(loader)[0].numpy()
        X = X.reshape(X.shape[0], -1)
        scores = 1 - self.model.predict(X)
        return self.moving_average(scores, self.bs_seq)

    def fit(self):
        X = self.get_data(self.train_loader_seq)[0].numpy()
        X = X.reshape(X.shape[0], -1)
        self.model.fit(X)