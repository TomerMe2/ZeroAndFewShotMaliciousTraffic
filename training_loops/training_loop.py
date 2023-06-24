from abc import ABC, abstractmethod


class TrainingLoop(ABC):
    
    @abstractmethod
    def fit(self, train_dataset, test_dataset, batch_size, num_workers):
        pass
    
    @staticmethod
    @abstractmethod
    def load_model(path):
        pass
