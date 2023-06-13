import hydra
from omegaconf import DictConfig
import numpy as np
from clearml import Task
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl


@hydra.main(version_base='1.3', config_path="configs/", config_name="train.yaml")
def main(conf: DictConfig):
    conf = hydra.utils.instantiate(conf)
    task = Task.init(
        project_name="few_shot_malicious_traffic", task_name=conf.experiment_name
    )
    
    train_dataset = conf.dataset(is_train=True, include_malicious_traffic=False, lbl_is_src_ip=True)
    test_dataset = conf.dataset(is_train=False, include_malicious_traffic=False, lbl_is_src_ip=True)
    
    model = conf.model(input_size=train_dataset.row_size)
    training_loop = conf.training_loop(model=model, num_classes=train_dataset.num_classes)
    
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
    
    trainer = pl.Trainer()
    trainer.fit(training_loop, train_dataloader, test_dataloader)
    

if __name__ == '__main__':
    main()