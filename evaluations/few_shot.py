import os
import numpy as np
import torch
import random
import statistics
from clearml import Task, Logger
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt

from evaluations.evaluation import Evaluation



class FewShotEevaluation(Evaluation): # similar to ero shit, with diffrent memorization
    
    def __init__(self, model, is_neural_network, out_path, MEM_SIZES, REPEAT_PER_MEM,fprs_wp):
        super().__init__(model, is_neural_network, out_path)
        self.MEM_SIZES = MEM_SIZES # ammout of memory size to test
        self.REPEAT_PER_MEM = REPEAT_PER_MEM # ammout of random selection to make for each memory
        self.fprs_wp = fprs_wp # fapse positive rates to display


    def evaluate(self, train_dataloader, test_dataloader, test_dataset):

        #memorization & scores infering
        embs_memory = self.memorize(train_dataloader)
        all_score_for_being_malicious_on_benign_flows, all_score_for_being_malicious_on_malicious_flows, all_malicious_attack_labels = self.infer(test_dataloader, embs_memory, test_dataset.benign_label)      


        # fir each mem size averdge metrics, and repurt graph
        for mem_size, mem_score_for_being_malicious_on_benign_flows, mem_score_for_being_malicious_on_malicious_flows, mem_malicious_attack_labels in tqdm(zip(self.MEM_SIZES, all_score_for_being_malicious_on_benign_flows, all_score_for_being_malicious_on_malicious_flows, all_malicious_attack_labels)):

            list_fprs, list_tprs, list_auc = [], [], [] # list of all attempts' scores fir this mem size

            for random_attempt, score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels in zip(range(self.REPEAT_PER_MEM), mem_score_for_being_malicious_on_benign_flows, mem_score_for_being_malicious_on_malicious_flows, mem_malicious_attack_labels) :

                fprs, tprs, thresholds, auc_score = self.calc_roc(score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows)
                list_fprs.append(fprs)
                list_tprs.append(tprs)
                list_auc.append(auc_score)

            # calculating metrics averages and stds
            mean_fprs = [statistics.mean(coresponding_metrics) for coresponding_metrics in zip(*list_fprs)]
            mean_tprs = [statistics.mean(coresponding_metrics) for coresponding_metrics in zip(*list_tprs)]
            mean_auc = statistics.mean(auc)
            std_fprs = [statistics.stdev(coresponding_metrics) for coresponding_metrics in zip(*list_fprs)]
            std_tprs = [statistics.stdev(coresponding_metrics) for coresponding_metrics in zip(*list_tprs)]
            std_auc = statistics.stdev(auc)

            #draw ruc curve for aveged metrics
            self.plot_roc(mean_fprs, mean_tprs, [0]*len(mean_fprs), mean_auc, title='Few Shot Evaluation ROC Curve', filename=f'MEM{mem_size}_roc_auc_of_{mean_auc:.3f}.csv')
        
        
        
        
        
        # self.compute_metrics_per_class_at_working_points(score_for_being_malicious_on_malicious_flows,
        #                                                  test_dataset.label_encoder.inverse_transform(malicious_attack_labels), [0.0001, 0.001, 0.01, 0.05, 0.1],
        #                                                  fprs, tprs, thresholds)


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
        batch = 0                                       # a counter of the batchID. at end holds |batches|
        test_embedings = {}                             # key : batchID, value: list of embedings of the batch
        test_labels = {}                                # key : batchID, value: list of labels of the batch  
        embs_memory =list(embs_memory.values())         # list of all test memirized embedings by attack type
        embs_memory_labels =list(embs_memory.keys())    # list of curesponding atttack types    
        
        if not self.is_neural_network:
            raise Exception('few shot is suported for DL.')
        

        # -----------------------------------------------------------------------


        # calculating embedings for each batch
        for  x, y in tqdm(test_dataloader):
            y = y.cpu().numpy()      
            test_embedings[batch] = self.model(x)
            test_labels[batch] = y

            batch +=1


        # -----------------------------------------------------------------------


        score_for_being_malicious_on_benign_flows = []
        score_for_being_malicious_on_malicious_flows = []
        malicious_attack_labels = []

        # for each memory size, calculate metrics @REPEAT_PER_MEM times
        for mem_size in  tqdm(self.MEM_SIZES):

            mem_score_for_being_malicious_on_benign_flows = []
            mem_score_for_being_malicious_on_malicious_flows = []
            mem_malicious_attack_labels = []

            for random_attempt in range(self.REPEAT_PER_MEM) :
                cur_embs_memory = [np.stack(self._pickN(mem, mem_size)).mean(axis=0) for mem in embs_memory]

                attempt_score_for_being_malicious_on_benign_flows = []
                attempt_score_for_being_malicious_on_malicious_flows = []
                attempt_malicious_attack_labels = []

                for batchID in range(batch) :
                    sims = cosine_similarity(test_embedings[batchID], cur_embs_memory)

                    max_sims = []
                    for s in sims :
                        
                        # check all labels, finding the most similar mean embedding from memorization
                        best_matching_score = float('-inf')
                        best_matching_label = None
                        for label, similarity in zip(embs_memory_labels, s):
                            if similarity > best_matching_score :
                                best_matching_score = similarity
                                best_matching_label = label

                        # if the label is bengin - we the surer we are this is bengin, the **LOWER** the score
                        # if the label is malicious - we the surer we are this is malicious, the **HIGHER** the score
                        max_sims.append(1-best_matching_score if best_matching_label == benign_label else 1+best_matching_score)
           
                    scores = np.array(max_sims)

                    attempt_score_for_being_malicious_on_benign_flows.extend(scores[test_labels[batchID] == benign_label].tolist())
                    attempt_score_for_being_malicious_on_malicious_flows.extend(scores[test_labels[batchID] != benign_label].tolist())
                    attempt_malicious_attack_labels.extend(test_labels[batchID][test_labels[batchID] != benign_label].tolist())
          
                mem_score_for_being_malicious_on_benign_flows.append(attempt_score_for_being_malicious_on_benign_flows)
                mem_score_for_being_malicious_on_malicious_flows.append(attempt_score_for_being_malicious_on_malicious_flows)
                mem_malicious_attack_labels.append(attempt_malicious_attack_labels)
            

            score_for_being_malicious_on_benign_flows.append(mem_score_for_being_malicious_on_benign_flows)
            score_for_being_malicious_on_malicious_flows.append(mem_score_for_being_malicious_on_malicious_flows)
            malicious_attack_labels.append(mem_malicious_attack_labels)


        # -----------------------------------------------------------------------
        
        #returning all calculated scores
        return score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels





    # get a sublist of n random elements, if n is too large, return entire list
    def _pickN(self, lst, n):
        try:
            return random.sample(lst, n)
        except ValueError:
            return lst