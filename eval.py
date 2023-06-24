import importlib

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from clearml import Task


@hydra.main(version_base='1.3', config_path="configs", config_name="eval.yaml")
def main(conf: DictConfig):
    conf = hydra.utils.instantiate(conf)

    task = Task.init(
        project_name="few_shot_malicious_traffic", task_name=conf.experiment_name
    )
    
    train_dataset = conf.dataset(is_train=True, include_malicious_traffic=False, lbl_is_src_ip=True)
    test_dataset = conf.dataset(is_train=False, include_malicious_traffic=True, lbl_is_src_ip=False)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)  
    
    training_loop_class = getattr(importlib.import_module(conf.training_loop_class.rsplit('.', 1)[0]), conf.training_loop_class.rsplit('.', 1)[1])
    model = training_loop_class.load_model(conf.checkpoint_path)

    evaluation = conf.evaluation(model)
    evaluation.evaluate(train_dataloader, test_dataloader, test_dataset)


if __name__ == '__main__':
    main()