import torchmetrics

from losses.arc_face_loss_with_logits_out import ArcFaceLossWithLogitsOut
from training_loops.neural_network_training_loop import NeuralNetworkTrainingLoop


class ClassificationTrainingLoop(NeuralNetworkTrainingLoop):

    def __init__(self, model, num_classes):
        super().__init__(model)
        self.loss = ArcFaceLossWithLogitsOut(
            num_classes=num_classes, embedding_size=model.embedding_size)
        
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.save_hyperparameters()

    def step(self, batch, kind):
        x, y = batch
        embeddings = self.forward(x)
        loss, logits = self.loss(embeddings, y)
        acc = self.acc(logits, y)
        
        self.log(f'loss/{kind}', loss, on_step=False, on_epoch=True)
        self.log(f'acc/{kind}', acc, on_step=False, on_epoch=True)
        return loss

    @staticmethod
    def load_model(path):
        model = ClassificationTrainingLoop.load_from_checkpoint(path).model
        model.eval()
        return model
