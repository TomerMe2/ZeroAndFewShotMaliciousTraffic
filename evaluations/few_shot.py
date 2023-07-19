import numpy as np
import torch
import random
import statistics
from clearml import Logger
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd

from evaluations.evaluation import Evaluation



class FewShotEevaluation(Evaluation): # similar to ero shit, with diffrent memorization
    
    def __init__(self, model, is_neural_network, out_path, MEM_SIZES, REPEAT_PER_MEM,fprs_wp):
        super().__init__(model, is_neural_network, out_path)
        self.MEM_SIZES = MEM_SIZES # ammout of memory size to test
        self.REPEAT_PER_MEM = REPEAT_PER_MEM # ammout of random selection to make for each memory
        self.fprs_wp = fprs_wp # fapse positive rates to display


    def evaluate(self, train_dataloader, test_dataloader, test_dataset):

        #memorization & scores infering
        embs_memory = self.memorize(test_dataloader)
        all_score_for_being_malicious_on_benign_flows, all_score_for_being_malicious_on_malicious_flows, malicious_attack_labels = self.infer(test_dataloader, embs_memory, test_dataset.benign_label)      


        # fir each mem size averdge metrics, and repurt graph
        for mem_size, mem_score_for_being_malicious_on_benign_flows, mem_score_for_being_malicious_on_malicious_flows in tqdm(zip(self.MEM_SIZES, all_score_for_being_malicious_on_benign_flows, all_score_for_being_malicious_on_malicious_flows)):

            labels = np.unique(malicious_attack_labels)
            text_labels = test_dataset.label_encoder.inverse_transform(labels)
            
            # cache for graphs
            list_fprs, list_tprs, list_auc = [], [], [] # list of all attempts' scores fir this mem size

            #cache for tabels
            tprs_by_wp_label   = {} # key: (fpr_wp, label), value : list of the tprs of each attempt by the given fpr_wp
            global_tprs = {} # key: fpr_wp, value : list of the global tpr of each attempt by the given fpr_wp



            for random_attempt, score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows in zip(range(self.REPEAT_PER_MEM), mem_score_for_being_malicious_on_benign_flows, mem_score_for_being_malicious_on_malicious_flows) :

                fprs, tprs, thresholds, auc_score = self.calc_roc(score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows)
                list_fprs.append(fprs)
                list_tprs.append(tprs)
                list_auc.append(auc_score)

                for fpr_wp in self.fprs_wp:
                    idx = np.argmin(np.abs(fprs - fpr_wp))
                    threshold = thresholds[idx]
                    global_tpr = tprs[idx]

                    if fpr_wp not in global_tprs :
                        global_tprs[fpr_wp] = []
                    global_tprs[fpr_wp].append(global_tpr)

                    for curr_malicious_lbl in labels:
                        curr_malicious_scores = score_for_being_malicious_on_malicious_flows[malicious_attack_labels == curr_malicious_lbl]
                        [tpr, _] = self.metrics_given_class_and_threshold(threshold, curr_malicious_scores)
                        
                        if (fpr_wp, curr_malicious_lbl) not in tprs_by_wp_label :
                            tprs_by_wp_label[(fpr_wp, curr_malicious_lbl)] = []
                        tprs_by_wp_label[(fpr_wp, curr_malicious_lbl)].append(tpr)
                    

            # --------------------------------------------------

            # calculating metrics averages and stds
            mean_fprs = [statistics.mean(coresponding_metrics) for coresponding_metrics in zip(*list_fprs)]
            mean_tprs = [statistics.mean(coresponding_metrics) for coresponding_metrics in zip(*list_tprs)]
            mean_auc = statistics.mean(list_auc)

            #draw ruc curve for aveged metrics
            self.plot_roc(mean_fprs, mean_tprs, [0]*len(mean_fprs), mean_auc, title=f'Few Shot Evaluation ROC Curve @MEM{mem_size}', filename=f'MEM{mem_size}')

            #calculate tabels
            for fpr_wp in self.fprs_wp:
                mean_global_tpr = statistics.mean(global_tprs[fpr_wp])
                std_global_tpr = statistics.stdev(global_tprs[fpr_wp])

                metrics_per_class  = []
                for label, txt_label in zip(labels, text_labels):
                    tprs = tprs_by_wp_label[(fpr_wp, label)]
                    mean_tpr = statistics.mean(tprs)
                    std_tpr = statistics.mean(tprs)


                    metrics_per_class.append([txt_label, mean_tpr, 1-mean_tpr, std_tpr])
                
                curr_wp_metrics_df = pd.DataFrame(metrics_per_class,
                                              columns=['Malicious Class', 'tpr (rate of identified malicious flows)', 'fnr (rate of not identified malicious flow)', 'std of tpr'])
                
                Logger.current_logger().report_table(
                        "Binary Metrics Per Class at Working Point",
                        f"Metrics @ MEM={mem_size} FPR={fpr_wp}, TPR={mean_global_tpr} (std={std_global_tpr})",
                        iteration=0,
                        table_plot=curr_wp_metrics_df
                    )
                



    @torch.no_grad()
    def memorize(self, memorize_dataloader):
        if not self.is_neural_network:
            raise Exception('few shot is suported for DL.')
        
        embs_dct = {}    # {attack: [emb1, emb2 ...]}
        for x, y in memorize_dataloader:
            y = y.cpu().numpy()
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
        embs_memory_attacks =list(embs_memory.keys())   # list of curesponding atttack types    
        embs_memory =list(embs_memory.values())         # list of all test memirized embedings by attack type
        
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

        # labels are independent from mem size and fpr_wp, can be calculated seperetly once.
        for batchID in range(batch) :
            malicious_attack_labels.extend(test_labels[batchID][test_labels[batchID] != benign_label].tolist())
        malicious_attack_labels = np.array(malicious_attack_labels)

        # for each memory size, calculate metrics @REPEAT_PER_MEM times
        for mem_size in  tqdm(self.MEM_SIZES):

            mem_score_for_being_malicious_on_benign_flows = []
            mem_score_for_being_malicious_on_malicious_flows = []

            for random_attempt in range(self.REPEAT_PER_MEM) :
                cur_embs_memory = [np.stack(self._pickN(mem, mem_size)).mean(axis=0) for mem in embs_memory]

                attempt_score_for_being_malicious_on_benign_flows = []
                attempt_score_for_being_malicious_on_malicious_flows = []

                for batchID in range(batch) :
                    sims = cosine_similarity(test_embedings[batchID], cur_embs_memory)

                    max_sims = []
                    for s in sims :
                        
                        # check all labels, finding the most similar mean embedding from memorization
                        best_matching_score = float('-inf')
                        best_matching_label = None
                        for label, similarity in zip(embs_memory_attacks, s):
                            if similarity > best_matching_score :
                                best_matching_score = similarity
                                best_matching_label = label

                        # if the label is bengin - we the surer we are this is bengin, the **LOWER** the score
                        # if the label is malicious - we the surer we are this is malicious, the **HIGHER** the score
                        max_sims.append(1-best_matching_score if best_matching_label == benign_label else 1+best_matching_score)
           
                    scores = np.array(max_sims)

                    attempt_score_for_being_malicious_on_benign_flows.extend(scores[test_labels[batchID] == benign_label].tolist())
                    attempt_score_for_being_malicious_on_malicious_flows.extend(scores[test_labels[batchID] != benign_label].tolist())
          
                mem_score_for_being_malicious_on_benign_flows.append(np.array(attempt_score_for_being_malicious_on_benign_flows))
                mem_score_for_being_malicious_on_malicious_flows.append(np.array(attempt_score_for_being_malicious_on_malicious_flows))
            

            score_for_being_malicious_on_benign_flows.append(mem_score_for_being_malicious_on_benign_flows)
            score_for_being_malicious_on_malicious_flows.append(mem_score_for_being_malicious_on_malicious_flows)


        # -----------------------------------------------------------------------
        
        #returning all calculated scores
        return score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels





    # get a sublist of n random elements, if n is too large, return entire list
    def _pickN(self, lst, n):
        try:
            return random.sample(lst, n)
        except ValueError:
            return lst