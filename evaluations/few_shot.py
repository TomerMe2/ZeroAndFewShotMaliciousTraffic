import numpy as np
import torch
import random
import statistics
from clearml import Task, Logger
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
from sklearn.metrics import roc_curve
import pandas as pd

from evaluations.evaluation import Evaluation



class FewShotEevaluation(Evaluation): # similar to ero shit, with diffrent memorization
    
    def __init__(self, model, is_neural_network, out_path):
        super().__init__(model, is_neural_network, out_path)
        self.MEM_SIZES = [1,5,10,30,50,300] # ammout of memory size to test
        self.REPEAT_PER_MEM = 100 # ammout of random selection to make for each memory


    def evaluate(self, train_dataloader, test_dataloader, test_dataset):
        embs_memory = self.memorize(test_dataloader)
        all_score_for_being_malicious_on_benign_flows, all_score_for_being_malicious_on_malicious_flows, all_malicious_attack_labels = self.infer(test_dataloader, embs_memory, test_dataset.benign_label)      
        fprs_wp = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3]

        for mem_size, mem_score_for_being_malicious_on_benign_flows, mem_score_for_being_malicious_on_malicious_flows, mem_malicious_attack_labels in tqdm(zip(self.MEM_SIZES, all_score_for_being_malicious_on_benign_flows, all_score_for_being_malicious_on_malicious_flows, all_malicious_attack_labels)) :
            
            #keys : (fpr_wp, label)
            attempt_fnrs = {}
            attempt_tprs = {}
            
            tprs_metrics = {} #key : fpr_wp
            labels = np.unique(test_dataset.label_encoder.inverse_transform(mem_malicious_attack_labels[0]))

            for random_attempt, attempt_score_for_being_malicious_on_benign_flows, attempt_score_for_being_malicious_on_malicious_flows, attempt_malicious_attack_labels in zip(range(self.REPEAT_PER_MEM), mem_score_for_being_malicious_on_benign_flows, mem_score_for_being_malicious_on_malicious_flows, mem_malicious_attack_labels) :

                fprs, tprs, thresholds = roc_curve(['Malicious'] * len(attempt_score_for_being_malicious_on_malicious_flows) + ['Benign'] * len(attempt_score_for_being_malicious_on_benign_flows),
                                        attempt_score_for_being_malicious_on_malicious_flows.tolist() + attempt_score_for_being_malicious_on_benign_flows.tolist(),
                                        pos_label='Malicious')

            
                for fpr_wp in fprs_wp:
                    idx = np.argmin(np.abs(fprs - fpr_wp))
                    threshold = thresholds[idx]
                    tpr = tprs[idx]
                    
                    if fpr_wp not in tprs_metrics :
                        tprs_metrics[fpr_wp] = []
                    tprs_metrics[fpr_wp].append(tpr)


                    metrics_per_class = []
                    lbls = np.unique(test_dataset.label_encoder.inverse_transform(np.unique(attempt_malicious_attack_labels)))
                    for curr_malicious_lbl, lbl in zip(np.unique(attempt_malicious_attack_labels), lbls):
                        curr_malicious_scores = attempt_score_for_being_malicious_on_malicious_flows[attempt_malicious_attack_labels == curr_malicious_lbl]
                        attempt_tpr, attempt_fpr = self.metrics_given_class_and_threshold(threshold, curr_malicious_scores)
                        
                        if (fpr_wp, lbl) not in attempt_fnrs :
                            attempt_fnrs[(fpr_wp, lbl)] = []
                            attempt_tprs[(fpr_wp, lbl)] = []

                        attempt_fnrs[(fpr_wp, lbl)].append(attempt_fpr)
                        attempt_tprs[(fpr_wp, lbl)].append(attempt_tpr)


            for fpr_wp in fprs_wp:
                tpr = tprs_metrics[fpr_wp]
                tpr_mean = statistics.mean(tpr)
                tpr_std = statistics.stdev(tpr)

                metrics_per_class = []
                for label in labels:
                    fnrs = attempt_fnrs[(fpr_wp, label)]
                    tprs = attempt_tprs[(fpr_wp, label)]

                    metrics_per_class.append([label] + [statistics.mean(tprs), statistics.stdev(tprs),   statistics.mean(fnrs), statistics.stdev(fnrs)])

                curr_wp_metrics_df = pd.DataFrame(metrics_per_class,
                                              columns=['Malicious Class', 'tpr (rate of identified malicious flows)', 'std of tpr', 'fnr (rate of not identified malicious flow)', 'std of fnr'])
                
                Logger.current_logger().report_table(
                    "Binary Metrics Per Class at Working Point",
                    f"Metrics @ FPR={fpr_wp}, TPR={tpr_mean} (std={tpr_std}), MEM={mem_size}",
                    iteration=0,
                    table_plot=curr_wp_metrics_df
                )


    def metrics_given_class_and_threshold(self, threshold, curr_malicious_scores):
        true_positive = np.sum(curr_malicious_scores > threshold)
        false_negative = np.sum(curr_malicious_scores <= threshold)
            
        tpr = true_positive / (true_positive + false_negative)
        fnr = false_negative / (false_negative + true_positive)
            
        return tpr, fnr



    @torch.no_grad()
    def memorize(self, memorize_dataloader):
        if not self.is_neural_network:
            raise Exception('few shot is suported for DL.')
        
        embs_dct = {}    # {attack: [emb1, emb2 ...]}
        for x, y in memorize_dataloader:
            embeddings = self.model(x)
            
            for emb, label in zip(embeddings, y):
                label = label.item()

                if label not in embs_dct: # this is a new label, initalizing empty embeding list.
                    embs_dct[label] = []
                
                embs_dct[label].append(emb.cpu().numpy())
        
        return embs_dct



    @torch.no_grad()
    def infer(self, test_dataloader, embs_memory, benign_label):
        embs_memory_labels, embs_memory =list(embs_memory.keys()), list(embs_memory.values())
        numOfBatches = 0
        batch = 0
        all_attempt_scores = {} # key : (batchID, attemptID, mem_size), value : scores list
        ys = {} # key : batchID, value : y list

        for  x, y in tqdm(test_dataloader):
            y = y.cpu().numpy()                    
            embeddings = self.model(x)

            for mem_size in self.MEM_SIZES:

                for random_attempt in range(self.REPEAT_PER_MEM) :
                    cur_embs_memory = [np.stack(self.pickN(mem, mem_size)).mean(axis=0) for mem in embs_memory]        
                    sims = cosine_similarity(embeddings.cpu().numpy(), cur_embs_memory)

                    max_sims = []
                    for v in sims :
                            
                        best_matching_score = float('-inf')
                        best_matching_label = None
                        for label, similarity in zip(embs_memory_labels, v):
                            if similarity > best_matching_score :
                                best_matching_score = similarity
                                best_matching_label = label

                        max_sims.append(1-best_matching_score if best_matching_label == benign_label else 1+best_matching_score)
                        
                    all_attempt_scores[(batch, random_attempt, mem_size)] = np.array(max_sims)
                    if batch not in ys :
                        ys[batch] = y
           
          
            batch+=1
            numOfBatches = max(numOfBatches, batch)

        

        score_for_being_malicious_on_benign_flows = []
        score_for_being_malicious_on_malicious_flows = []
        malicious_attack_labels = []

        for mem_size in tqdm(self.MEM_SIZES):
            mem_score_for_being_malicious_on_benign_flows = []
            mem_score_for_being_malicious_on_malicious_flows = []
            mem_malicious_attack_labels = []

            for random_attempt in range(self.REPEAT_PER_MEM) :
                attempt_score_for_being_malicious_on_benign_flows = []
                attempt_score_for_being_malicious_on_malicious_flows = []
                attempt_malicious_attack_labels = []

                for batch in range(numOfBatches):
                    scores = all_attempt_scores[(batch, random_attempt, mem_size)]

                    attempt_score_for_being_malicious_on_benign_flows.extend(scores[ys[batch] == benign_label])
                    attempt_score_for_being_malicious_on_malicious_flows.extend(scores[ys[batch] != benign_label])
                    attempt_malicious_attack_labels.extend(ys[batch][ys[batch] != benign_label])

                mem_score_for_being_malicious_on_benign_flows.append(attempt_score_for_being_malicious_on_benign_flows)
                mem_score_for_being_malicious_on_malicious_flows.append(attempt_score_for_being_malicious_on_malicious_flows)
                mem_malicious_attack_labels.append(attempt_malicious_attack_labels)  

            score_for_being_malicious_on_benign_flows.append(np.array(mem_score_for_being_malicious_on_benign_flows))
            score_for_being_malicious_on_malicious_flows.append(np.array(mem_score_for_being_malicious_on_malicious_flows))
            malicious_attack_labels.append(np.array(mem_malicious_attack_labels))

        return np.array(score_for_being_malicious_on_benign_flows), np.array(score_for_being_malicious_on_malicious_flows), np.array(malicious_attack_labels)




    # get a sublist of n random elements, if n is too large, return entire list
    def pickN(self, lst, n):
        try:
            return random.sample(lst, n)
        except ValueError:
            return lst