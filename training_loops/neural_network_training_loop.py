
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from abc import abstractmethod, ABC

from training_loops.training_loop import TrainingLoop


class NeuralNetworkTrainingLoop(pl.LightningModule, TrainingLoop, ABC):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @abstractmethod
    def step(self, batch, kind):
        pass

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, 'train')

    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, 'val')
    
    def fit(self, train_dataset, test_dataset, batch_size, num_workers):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                      shuffle=True, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                     shuffle=False, num_workers=num_workers)
        
        trainer = pl.Trainer(max_epochs=50)
        trainer.fit(self, train_dataloader, test_dataloader)

