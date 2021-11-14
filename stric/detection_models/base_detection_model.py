from abc import ABC, abstractmethod

class BaseDetector(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def train(self):
        '''Train detector (if based on prediction error methods train the predictive model and the dectector)'''
        pass

    @abstractmethod
    def get_anomaly_scores(self):
        '''Get anomaly scores for the dataset'''
        pass

    @abstractmethod
    def visualize(self):
        '''Visualize anomaly scores and time series predictions (if present)'''
        pass