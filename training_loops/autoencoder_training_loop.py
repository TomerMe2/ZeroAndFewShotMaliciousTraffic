from training_loops.neural_network_training_loop import NeuralNetworkTrainingLoop
import torch.nn as nn


class AutoEncoderTrainingLoop(NeuralNetworkTrainingLoop):

    def __init__(self, model, num_classes):
        super().__init__(model)
        self.loss =nn.MSELoss()
        self.save_hyperparameters(ignore=['model'])

    def step(self, batch, kind):
        x, _ = batch
        embeddings = self.forward(x)
        loss = self.loss(embeddings, x)
 
        self.log(f'loss/{kind}', loss, on_step=False, on_epoch=True)
        return loss

    @staticmethod
    def load_model(path):
        model = AutoEncoderTrainingLoop.load_from_checkpoint(path).model
        model.eval()
        return model
