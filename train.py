import hydra
from omegaconf import DictConfig
from clearml import Task


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
    training_loop.fit(train_dataset, test_dataset,
                      conf.batch_size, conf.num_workers)    


if __name__ == '__main__':
    main()