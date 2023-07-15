import numpy as np
import torch
import random
import statistics
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from evaluations.evaluation import Evaluation



class FewShotEevaluation(Evaluation): # similar to ero shit, with diffrent memorization
    
    def __init__(self, model, is_neural_network, out_path):
        super().__init__(model, is_neural_network, out_path)
        self.MEM_SIZES = [1,5,10,30,50,300] # ammout of memory size to test
        self.REPEAT_PER_MEM = 1 #30 # ammout of random selection to make for each memory


    def evaluate(self, train_dataloader, test_dataloader, test_dataset):
        embs_memory = self.memorize(test_dataloader)
        score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels = self.infer(test_dataloader, embs_memory, test_dataset.benign_label)
        
        for mem_size, score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels in zip(self.MEM_SIZES, score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels) :
        
            self.draw_histogram(score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, title=f'Cosine Similarity Histogram of Benign and Malicious Flows @MEM{mem_size}')
            fprs, tprs, thresholds = self.draw_roc(score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, title=f'Few Shot Evaluation ROC Curve @MEM{mem_size}', filename=f'MEM{mem_size}_roc_auc')



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
        score_for_being_malicious_on_benign_flows = []
        score_for_being_malicious_on_malicious_flows = []
        malicious_attack_labels = []

        embs_memory_labels, embs_memory =list(embs_memory.keys()), list(embs_memory.values())


        for mem_size in self.MEM_SIZES:
            cur_embs_memory = [np.stack(self.pickN(mem, mem_size)).mean(axis=0) for mem in embs_memory]

            cur_score_for_being_malicious_on_benign_flows = []
            cur_score_for_being_malicious_on_malicious_flows = []
            cur_malicious_attack_labels = []
            for  x, y in tqdm(test_dataloader):
                y = y.cpu().numpy()
                
                embeddings = self.model(x)

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
                
                scores = np.array(max_sims)

                cur_score_for_being_malicious_on_benign_flows.extend(np.array(scores[y == benign_label].tolist()))
                cur_score_for_being_malicious_on_malicious_flows.extend(np.array(scores[y != benign_label].tolist()))
                cur_malicious_attack_labels.extend(np.array(y[y != benign_label].tolist()))

            score_for_being_malicious_on_benign_flows.append(np.array(cur_score_for_being_malicious_on_benign_flows))
            score_for_being_malicious_on_malicious_flows.append(np.array(cur_score_for_being_malicious_on_malicious_flows))
            malicious_attack_labels.append(np.array(cur_malicious_attack_labels))
            
        return np.array(score_for_being_malicious_on_benign_flows), np.array(score_for_being_malicious_on_malicious_flows), np.array(malicious_attack_labels)




    # get a sublist of n random elements, if n is too large, return entire list
    def pickN(self, lst, n):
        try:
            return random.sample(lst, n)
        except ValueError:
            return lst






















































    # overriding the evaluate method since whe nned to memorize *test* partial data, nit train 
    # def evaluate(self, train_dataloader, test_dataloader, test_dataset):
    #     embs_memory = self.memorize(test_dataloader)
    #     score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels = self.infer(test_dataloader, embs_memory, test_dataset.benign_label)
            
    #     for mem_size, score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels in zip(self.MEM_SIZES, score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels) :                      
    #         mean_scores_benign = [score_std[0] for score_std in score_for_being_malicious_on_benign_flows]
    #         std_scores_benign =[score_std[1] for score_std in score_for_being_malicious_on_benign_flows]

    #         mean_scores_malicious = [score_std[0] for score_std in score_for_being_malicious_on_malicious_flows]
    #         std_scores_malicious = [score_std[1] for score_std in score_for_being_malicious_on_malicious_flows]
            
                               
    #         self.draw_histogram(mean_scores_benign, mean_scores_malicious, title=f'Cosine Similarity Histogram of Benign and Malicious Flows @MEM{mem_size}, bengin std={std_scores_benign}, malicious std={std_scores_malicious}')
    #         fprs, tprs, thresholds = self.draw_roc(mean_scores_benign, mean_scores_malicious, title=f'Few Shot Evaluation ROC Curve @MEM{mem_size}, bengin std={std_scores_benign}, malicious std={std_scores_malicious}', filename=f'MEM{mem_size}_roc_auc')
    #         #self.compute_metrics_per_class_at_working_points(score_for_being_malicious_on_malicious_flows,
    #         #                                             test_dataset.label_encoder.inverse_transform(malicious_attack_labels), [0.0001, 0.001, 0.01, 0.05, 0.1],
    #         #                                             fprs, tprs, thresholds)
       

    # @torch.no_grad()
    # def memorize(self, memorize_dataloader):
    #     if not self.is_neural_network:
    #         raise Exception('few shot is suported for DL.')
        
    #     embs_dct = {}    # {attack: [emb1, emb2 ...]}
    #     for batch in memorize_dataloader:
    #         x, y = batch
    #         embeddings = self.model(x)
            
    #         for emb, label in zip(embeddings, y):
    #             label = label.item()

    #             if label not in embs_dct: # this is a new label, initalizing empty embeding list.
    #                 embs_dct[label] = []
                
    #             embs_dct[label].append(emb.cpu().numpy())
        
    #     return embs_dct

    


    # @torch.no_grad()
    # def infer(self, test_dataloader, embs_memory, benign_label):
    #     score_for_being_malicious_on_benign_flows = []
    #     score_for_being_malicious_on_malicious_flows = []
    #     malicious_attack_labels = []

    #     for batch in tqdm(test_dataloader):
    #         x, y = batch
    #         y = y.cpu().numpy()
            
    #         embeddings = self.model(x)
    #         scores = []

    #         for emb in embeddings:                            
    #             scores_per_size = []
    #             for mem_size in self.MEM_SIZES :
                        
    #                 scores_per_random_attempt = []
    #                 for attempt in range(self.REPEAT_PER_MEM) :
                            
    #                     best_matching_score = float('-inf')
    #                     best_matching_label = None
    #                     for test_label, test_embeding in embs_memory.items() :

    #                         test_mean = np.stack(self.pickN(test_embeding, mem_size)).mean(axis=0) 
    #                         current_similarity = cosine_similarity(emb.reshape(1, -1), test_mean.reshape(1, -1))[0][0]

    #                         if current_similarity > best_matching_score :
    #                             best_matching_score = current_similarity
    #                             best_matching_label = test_label
                            
    #                     # for benign_label we ant low score, for attack we want big score
    #                     scores_per_random_attempt.append(1-best_matching_score if best_matching_label == benign_label else 1+best_matching_score)
                        
    #                 scores_per_size.append((statistics.mean(scores_per_random_attempt), statistics.stdev(scores_per_random_attempt)))
               
    #             scores.append(scores_per_size)
            
            
    #         scores = torch.tensor(scores)
    #         score_for_being_malicious_on_benign_flows.append(np.array(scores[y == benign_label].tolist()))
    #         score_for_being_malicious_on_malicious_flows.append(np.array(scores[y != benign_label].tolist()))
    #         malicious_attack_labels.append(np.array(y[y != benign_label].tolist()))
            
    #     return np.array(score_for_being_malicious_on_benign_flows), np.array(score_for_being_malicious_on_malicious_flows), np.array(malicious_attack_labels)