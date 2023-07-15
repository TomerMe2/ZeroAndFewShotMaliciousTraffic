import numpy as np
import torch
import random
import statistics
from clearml import Task
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

from evaluations.evaluation import Evaluation



class FewShotEevaluation(Evaluation): # similar to ero shit, with diffrent memorization
    
    def __init__(self, model, is_neural_network, out_path):
        super().__init__(model, is_neural_network, out_path)
        self.MEM_SIZES = [300] #[1,5,10,30,50,300] # ammout of memory size to test
        self.REPEAT_PER_MEM = 30 # ammout of random selection to make for each memory


    def evaluate(self, train_dataloader, test_dataloader, test_dataset):
        embs_memory = self.memorize(test_dataloader)
        score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels, stds = self.infer(test_dataloader, embs_memory, test_dataset.benign_label)
        
        for mem_size, score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels, stds in zip(self.MEM_SIZES, score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels, stds) :
            std = statistics.mean(stds)

            # upload std vector
            filename = f'stds_MEM_{mem_size}.txt'
            save_path = os.path.join(self.out_path, filename)
            with open(save_path, 'w') as file:
                for item in stds:
                    file.write(str(item) + '\n')
            Task.current_task().upload_artifact(filename, artifact_object=save_path)

        
            self.draw_histogram(score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, title=f'Cosine Similarity Histogram of Benign and Malicious Flows @MEM{mem_size}, std={std}')
            fprs, tprs, thresholds = self.draw_roc(score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, title=f'Few Shot Evaluation ROC Curve @MEM{mem_size}, std={std}', filename=f'MEM{mem_size}_roc_auc')



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
        scores_stds = []

        for mem_size in self.MEM_SIZES:
            mem_score_for_being_malicious_on_benign_flows = []
            mem_score_for_being_malicious_on_malicious_flows = []
            mem_malicious_attack_labels = []
            stds = []

            for batch in range(numOfBatches):
                attempt_scores = []
                for attempt in range(self.REPEAT_PER_MEM) :
                    attempt_scores.append(all_attempt_scores[(batch, attempt, mem_size)])
                attempt_scores = list(zip(*attempt_scores))
                batch_score = np.array([statistics.mean(x_scores) for x_scores in attempt_scores])
                stds.extend([0 if len(x_scores) == 1 else statistics.stdev(x_scores) for x_scores in attempt_scores])

                mem_score_for_being_malicious_on_benign_flows.extend(batch_score[ys[batch] == benign_label])
                mem_score_for_being_malicious_on_malicious_flows.extend(batch_score[ys[batch] != benign_label])
                mem_malicious_attack_labels.extend(ys[batch][ys[batch] != benign_label])

            score_for_being_malicious_on_benign_flows.append(np.array(mem_score_for_being_malicious_on_benign_flows))
            score_for_being_malicious_on_malicious_flows.append(np.array(mem_score_for_being_malicious_on_malicious_flows))
            malicious_attack_labels.append(np.array(mem_malicious_attack_labels))
            scores_stds.append(np.array(stds))


        return np.array(score_for_being_malicious_on_benign_flows), np.array(score_for_being_malicious_on_malicious_flows), np.array(malicious_attack_labels), np.array(scores_stds)




    # get a sublist of n random elements, if n is too large, return entire list
    def pickN(self, lst, n):
        try:
            return random.sample(lst, n)
        except ValueError:
            return lst