from abc import ABC, abstractmethod
import os

import numpy as np
from clearml import Logger, Task
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class Evaluation(ABC):
    
    def __init__(self, model, is_neural_network, out_path):
        self.model = model
        self.is_neural_network = is_neural_network
        self.out_path = out_path
        
        if os.path.exists(self.out_path):
            if not os.path.isdir(self.out_path):
                raise ValueError(f"out_path {self.out_path} exists and is not a directory")
        else:
            os.mkdir(self.out_path)
                
    
    def evaluate(self, train_dataloader, test_dataloader, test_dataset):
        embs_memory = self.memorize(train_dataloader)
        score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, malicious_attack_labels = self.infer(test_dataloader, embs_memory, test_dataset.benign_label)
        self.draw_histogram(score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows)
        fprs, tprs, thresholds = self.draw_roc(score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows)
        self.compute_metrics_per_class_at_working_points(score_for_being_malicious_on_malicious_flows,
                                                         test_dataset.label_encoder.inverse_transform(malicious_attack_labels), [0.0001, 0.001, 0.01, 0.05, 0.1],
                                                         fprs, tprs, thresholds)

    @abstractmethod
    def memorize(self, memorize_dataloader):
        """
        returns a dict of {key: mean_embedding}
        like, {src_ip1: mean_emb, src_ip2: mean_emb, ...}
        or {attack_type1: mean_emb, attack_type2: mean_emb, ...}
        """
        pass
    
    @abstractmethod
    def infer(self, test_dataloader, embs_memory, benign_label):
        pass
    
    def compute_metrics_per_class_at_working_points(self, score_for_being_malicious_on_malicious_flows,
                                                    malicious_attack_labels, fprs_working_points, fprs, tprs, thresholds, title_extension=''):
                  
        metrics_per_class = {}
        for fpr_wp in fprs_working_points:
            idx = np.argmin(np.abs(fprs - fpr_wp))
            threshold = thresholds[idx]
            
            metrics_per_class = []
            for curr_malicious_lbl in np.unique(malicious_attack_labels):
                curr_malicious_scores = score_for_being_malicious_on_malicious_flows[malicious_attack_labels == curr_malicious_lbl]
                metrics_for_curr_malicious = self.metrics_given_class_and_threshold(threshold, curr_malicious_scores)
                metrics_per_class.append([curr_malicious_lbl] + metrics_for_curr_malicious)
            
            curr_wp_metrics_df = pd.DataFrame(metrics_per_class,
                                              columns=['Malicious Class', 'tpr (rate of identified malicious flows)', 'fnr (rate of not identified malicious flow)'])
            Logger.current_logger().report_table(
                "Binary Metrics Per Class at Working Point",
                f"Metrics @ FPR={fpr_wp}, TPR={tprs[idx]}" + title_extension,
                iteration=0,
                table_plot=curr_wp_metrics_df
            )

    def metrics_given_class_and_threshold(self, threshold, curr_malicious_scores):
            true_positive = np.sum(curr_malicious_scores > threshold)
            false_negative = np.sum(curr_malicious_scores <= threshold)
            
            tpr = true_positive / (true_positive + false_negative)
            fnr = false_negative / (false_negative + true_positive)
            
            return [tpr, fnr]
                    
    def draw_histogram(self, score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, title='Cosine Similarity Histogram of Benign and Malicious Flows'):
            
        plt.hist(score_for_being_malicious_on_benign_flows, bins=100,
                 alpha=0.5, label='benign', density=True)
        plt.hist(score_for_being_malicious_on_malicious_flows, bins=100,
                 alpha=0.5, label='malicious', density=True)
        
        plt.title(title)
        plt.xlabel('Score of Being Malicious')
        plt.ylabel('Density')
        plt.legend()
        Logger.current_logger().report_matplotlib_figure(
            title=title,
            series="plot", iteration=0, figure=plt, report_interactive=False)
        plt.clf()

    
    def draw_roc(self, score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows, title='Zero Shot Evaluation ROC Curve', filename=None):
        fpr, tpr, thresholds, auc_score  = self.calc_roc(score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows)
        self.plot_roc(fpr, tpr, thresholds, auc_score, title, filename)

        return fpr, tpr, thresholds
    
    
    def plot_roc(self, fpr, tpr, thresholds, auc_score, title, filename):
        plt.plot(fpr, tpr)
        plt.title(title)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        Logger.current_logger().report_matplotlib_figure(
            title=title,
            series="plot", iteration=0, figure=plt, report_interactive=False)
        plt.clf()
          
        df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
        df = df.iloc[::100, :]   # the size of the df is too big, so we sample it
        filename =  f'{filename if filename else "roc_auc_of"}_{auc_score:.3f}.csv'
        save_path = os.path.join(self.out_path, filename)
        df.to_csv(save_path, index=False)
        Task.current_task().upload_artifact(filename, artifact_object=save_path)
    

    def calc_roc(self, score_for_being_malicious_on_benign_flows, score_for_being_malicious_on_malicious_flows):
        fpr, tpr, thresholds = roc_curve(['Malicious'] * len(score_for_being_malicious_on_malicious_flows) + ['Benign'] * len(score_for_being_malicious_on_benign_flows),
                                        score_for_being_malicious_on_malicious_flows.tolist() + score_for_being_malicious_on_benign_flows.tolist(),
                                        pos_label='Malicious')
        auc_score = auc(fpr, tpr)

        return fpr, tpr, thresholds, auc_score


       
    
