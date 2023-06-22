
import torch
import pytorch_lightning as pl
import torchmetrics

from losses.arc_face_loss_with_logits_out import ArcFaceLossWithLogitsOut


class ClassificationTrainingLoop(pl.LightningModule):

    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.loss = ArcFaceLossWithLogitsOut(
            num_classes=num_classes, embedding_size=model.embedding_size)
        
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def step(self, batch, kind):
        x, y = batch
        embeddings = self.forward(x)
        loss, logits = self.loss(embeddings, y)
        acc = self.acc(logits, y)
        
        self.log(f'loss/{kind}', loss, on_step=False, on_epoch=True)
        self.log(f'acc/{kind}', acc, on_step=False, on_epoch=True)
        return loss

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, 'train')

    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, 'val')
