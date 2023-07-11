import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn


from evaluations.evaluation import Evaluation


class AutoEvaluation(Evaluation):
    
    def __init__(self, model, is_neural_network, out_path):
        super().__init__(model, is_neural_network, out_path)
        self.loss =nn.MSELoss()

    @torch.no_grad()
    def memorize(self, memorize_dataloader):
       return None

    @torch.no_grad()
    def infer(self, test_dataloader, embs_memory, benign_label):
        score_for_being_malicious_on_benign_flows = []
        score_for_being_malicious_on_malicious_flows = []
        malicious_attack_labels = []
        for batch in tqdm(test_dataloader):
            x, y = batch
            y = y.cpu().numpy()

            recreated = self.model(x)
            scores = self.loss(x, recreated)
                
            score_for_being_malicious_on_benign_flows.extend(scores[y == benign_label].tolist())
            score_for_being_malicious_on_malicious_flows.extend(scores[y != benign_label].tolist())
            malicious_attack_labels.extend(y[y != benign_label].tolist())
            
        return np.array(score_for_being_malicious_on_benign_flows), np.array(score_for_being_malicious_on_malicious_flows), np.array(malicious_attack_labels)