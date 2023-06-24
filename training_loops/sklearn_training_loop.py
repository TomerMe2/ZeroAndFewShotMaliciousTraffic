import pickle

from sklearn.svm import OneClassSVM

from training_loops.training_loop import TrainingLoop


class SklearnTrainingLoop(TrainingLoop):
    
    def __init__(self, model, num_classes, model_save_path):
        super().__init__()
        self.model = model.model
        self.model_save_path = model_save_path
    
    def fit(self, train_dataset, test_dataset, batch_size, num_workers):
        X_train = train_dataset.X
        y_train = train_dataset.y

        self.model.fit(X_train, y_train)
        
        with open(self.model_save_path, 'wb') as fd:
            pickle.dump(self.model, fd)
    
    @staticmethod
    def load_model(path):
        with open(path, 'rb') as fd:
            return pickle.load(fd)

