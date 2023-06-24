from sklearn.svm import OneClassSVM as SklearnOneClassSVM


class OneClassSVM:
    
    def __init__(self, input_size):
        self.model = SklearnOneClassSVM()