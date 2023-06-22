import hydra
from omegaconf import DictConfig
import numpy as np
from clearml import Task, Logger
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pyrootutils
from sklearn.metrics import roc_curve
from tqdm import tqdm


pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)


@torch.no_grad()
def get_means_of_src_ips(model, memorize_dataloader):
    embs_dct = {}    # {src_ip: [emb1, emb2, ...]}
    for batch in memorize_dataloader:
        x, y = batch
        embeddings = model(x)
        
        for emb, src_ip in zip(embeddings, y):
            src_ip = src_ip.item()
            if src_ip not in embs_dct:
                embs_dct[src_ip] = []
            embs_dct[src_ip].append(emb.cpu().numpy())
    
    emb_means = [np.stack(embs).mean(axis=0) for embs in embs_dct.values()]  
    return emb_means


def draw_histogram(max_cosine_sims_benign, max_cosine_sims_malicious):
        
    plt.hist(max_cosine_sims_benign, bins=100, alpha=0.5, label='benign', density=True)
    plt.hist(max_cosine_sims_malicious, bins=100, alpha=0.5, label='malicious', density=True)
    
    plt.title('Cosine Similarity Histogram of Benign and Malicious Flows')
    plt.xlabel('Cosine Similarity To Some Known Source IP Center')
    plt.ylabel('Density')
    plt.legend()
    Logger.current_logger().report_matplotlib_figure(
        title="Cosine Similarity Histogram of Benign and Malicious Flows",
        series="plot", iteration=0, figure=plt, report_interactive=False)
    plt.clf()


@torch.no_grad()
def infer_zero_shot(model, test_dataloader, emb_means, benign_label):
    max_cosine_sims_benign = []
    max_cosine_sims_malicious = []
    for batch in tqdm(test_dataloader):
        x, y = batch
        y = y.cpu().numpy()
        embeddings = model(x)
        
        sims = cosine_similarity(embeddings.cpu().numpy(), emb_means)
        max_sims = sims.max(axis=1)
        max_cosine_sims_benign.extend(max_sims[y == benign_label].tolist())
        max_cosine_sims_malicious.extend(max_sims[y != benign_label].tolist())
        
    return np.array(max_cosine_sims_benign), np.array(max_cosine_sims_malicious)


def draw_roc(max_cosine_sims_benign, max_cosine_sims_malicious):
    malicious_score = 2 - max_cosine_sims_malicious
    benign_score = 2 - max_cosine_sims_benign
    fpr, tpr, thresholds = roc_curve(['Malicious'] * len(max_cosine_sims_malicious) + ['Benign'] * len(max_cosine_sims_benign),
                                     malicious_score.tolist() + benign_score.tolist(),
                                     pos_label='Malicious')
    plt.plot(fpr, tpr)
    plt.title('Zero Shot Evaluation ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    Logger.current_logger().report_matplotlib_figure(
        title="Zero Shot Evaluation ROC Curve",
        series="plot", iteration=0, figure=plt, report_interactive=False)
    plt.clf()

@hydra.main(version_base='1.3', config_path="../configs/", config_name="zero_shot_eval.yaml")
def main(conf: DictConfig):
    conf = hydra.utils.instantiate(conf)
    task = Task.init(
        project_name="few_shot_malicious_traffic", task_name=conf.experiment_name
    )
    
    train_dataset = conf.dataset(is_train=True, include_malicious_traffic=False, lbl_is_src_ip=True)
    test_dataset = conf.dataset(is_train=False, include_malicious_traffic=True, lbl_is_src_ip=False)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)  
    
    model = conf.model(input_size=test_dataset.row_size)
    training_loop = conf.training_loop(model=model, num_classes=train_dataset.num_classes)
    model = type(training_loop).load_from_checkpoint(conf.checkpoint_path).model
    model.eval()
    
    emb_means = get_means_of_src_ips(model, train_dataloader)
    max_cosine_sims_benign, max_cosine_sims_malicious = infer_zero_shot(model, test_dataloader, emb_means, test_dataset.benign_label)
    draw_histogram(max_cosine_sims_benign, max_cosine_sims_malicious)
    draw_roc(max_cosine_sims_benign, max_cosine_sims_malicious)


if __name__ == '__main__':
    main()