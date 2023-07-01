import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from evaluations.evaluation import Evaluation


class ZeroShotEevaluation(Evaluation):
    
    def __init__(self, model, is_neural_network, out_path):
        super().__init__(model, is_neural_network, out_path)

    @torch.no_grad()
    def memorize(self, memorize_dataloader):
        if not self.is_neural_network:
            # for classical machine learning we don't memorize
            return None
        
        embs_dct = {}    # {src_ip: [emb1, emb2, ...]}
        for batch in memorize_dataloader:
            x, y = batch
            embeddings = self.model(x)
            
            for emb, src_ip in zip(embeddings, y):
                src_ip = src_ip.item()
                if src_ip not in embs_dct:
                    embs_dct[src_ip] = []
                embs_dct[src_ip].append(emb.cpu().numpy())
        
        emb_means = [np.stack(embs).mean(axis=0) for embs in embs_dct.values()]  
        return emb_means

    @torch.no_grad()
    def infer(self, test_dataloader, embs_memory, benign_label):
        score_for_being_malicious_on_benign_flows = []
        score_for_being_malicious_on_malicious_flows = []
        malicious_attack_labels = []
        for batch in tqdm(test_dataloader):
            x, y = batch
            y = y.cpu().numpy()
            
            if self.is_neural_network:
                embeddings = self.model(x)
                
                sims = cosine_similarity(embeddings.cpu().numpy(), embs_memory)
                max_sims = sims.max(axis=1)
                scores = 1 - max_sims
                
            else:
                # decision function returns negative for outliers and positive for inliers
                # we wish to have positive scores for outliers and negative for inliers to be consistent with cosine distance
                # (bigger value -> bigger chance of being malicious)
                scores = -1 * self.model.decision_function(x.cpu().numpy())
                
            score_for_being_malicious_on_benign_flows.extend(scores[y == benign_label].tolist())
            score_for_being_malicious_on_malicious_flows.extend(scores[y != benign_label].tolist())
            malicious_attack_labels.extend(y[y != benign_label].tolist())
            
        return np.array(score_for_being_malicious_on_benign_flows), np.array(score_for_being_malicious_on_malicious_flows), np.array(malicious_attack_labels)