# Learn to Detect Attacks Without Seeing Attacks

Machine learning (ML) has demonstrated its effectiveness in identifying malicious network flows, addressing the core problem of network intrusion detection systems (NIDS). However, existing NIDS algorithms rely on training datasets that include known attacks, introducing a bias towards detecting similar zero-day attacks. For instance, attacks such as SYN flooding and port scanning may exhibit similar attributes like the number of packets per second or average packet size, leading to inherent biases when training on one known attack and testing on another. To mitigate this issue, we propose a method that enables the detection of attacks without the need to observe them during the training phase.

Our method aims to address two main scenarios in attack detection:
1. **Zero-shot detection of unseen attacks:** In this scenario, our algorithm operates autonomously after the training phase, capable of detecting attacks it has never encountered before, thereby reducing reliance on a pre-defined set of known attacks.
2. **Few-shot detection of manually annotated attacks:** In this scenario, a security analyst identifies a suspicious network flow and manually annotates it for further investigation. Subsequently, our algorithm leverages this limited set of annotated flows to discover additional suspicious network flows exhibiting similar behavior, effectively assisting the analyst in identifying potential threats.

A full report can be found [here](https://github.com/TomerMe2/ZeroAndFewShotMaliciousTraffic/blob/main/report.pdf).

## MetFlowGuard
MetaFlowGuard is a neural network model trained using AAMSoftmax with the primary objective of classifying the source IP of networking flows. The model's workflow consists of three stages:

1. **Training:** As discussed earlier, the neural network is trained with AAMSoftmax loss to classify source IP flows.
2. **Memorization:** In this stage, we generate the mean embedding of each source IP in the zero-shot scenario and the mean embedding of each attack type in the few-shot scenario and memorize them.
3. **Inference:** For each incoming flow, we calculate its embedding and determine its proximity to the memorized embeddings. In the zero-shot scenario, if it's close to the memorized embeddings, it's considered benign; otherwise, it's deemed malicious since we memorize benign flows. In the few-shot scenario, if it's close to a malicious memorized mean embedding, it's classified as malicious. If it's close to the benign memorized mean embedding, it's classified as benign;

For a visual representation of this workflow, please refer to Figure 6 in the report.



## Zero-Shot Evaluation
Each value in the table represents TPR (True Positive Rate) at a specific FPR (False Positive Rate) value, denoted as TPR@FPR=$x$.
(For example, MetaFlowGuard @0.01 will show that TPR of MetaFlowGuard algorithm at a FPR of 0.01)

### IoTID20 dataset
| Attack              | MetaFlowGuard @0.01 | MetaFlowGuard @0.001 | One-Class SVM @0.01 | One-Class SVM @0.001 | Autoencoder @0.01 | Autoencoder @0.001 |
|---------------------|--------------------|---------------------|--------------------|----------------------|-------------------|--------------------|
| DoS-Synflooding     | 0.99               | 0.91                | 0.99               | 0.99                 | 0.99              | 0.99               |
| MITM ARP Spoofing   | 0.25               | 0.17                | 0.17               | 0.00                 | 0.01              | 0.00               |
| Mirai-Ackflooding   | 0.68               | 0.30                | 0.54               | 0.00                 | 0.47              | 0.00               |
| Mirai-HTTP Flooding | 0.69               | 0.32                | 0.53               | 0.00                 | 0.47              | 0.00               |
| Mirai-Hostbruteforceg | 0.20             | 0.09                | 0.08               | 0.00                 | 0.07              | 0.01               |
| Mirai-UDP Flooding  | 0.90               | 0.31                | 0.86               | 0.55                 | 0.78              | 0.56               |
| Scan Hostport       | 0.08               | 0.03                | 0.04               | 0.00                 | 0.05              | 0.00               |
| Scan Port OS        | 0.04               | 0.00                | 0.00               | 0.00                 | 0.01              | 0.00               |
| All Attacks         | 0.58               | 0.28                | 0.50               | 0.27                 | 0.45              | 0.28               |

###  MQTT-IOT-IDS2020
| Attack             | MetaFlowGuard @0.05 | MetaFlowGuard @0.0001 |
|--------------------|---------------------|-----------------------|
| mqtt_bruteforce    | 0.53                | 0.00                  |
| scan_A             | 0.25                | 0.00                  |
| scan_sU            | 0.68                | 0.00                  |
| sparta             | 1.00                | 0.36                  |
| All Attacks        | 0.84                | 0.24                  |


## Few-Shot Evaluation
### IoTID20 dataset

| Attack                | 1-shot @0.25 | 1-shot @0.2 | 10-shot @0.25| 10-shot @0.2 | 30-shot @0.25 | 30-shot @0.2 |
|-----------------------|----------------------|------------------------|-----------------------|-------------------------|------------------------|-------------------------|
| DoS-Synflooding       | 0.84 $\pm$ 0.21      | 0.78 $\pm$ 0.21        | 0.97 $\pm$ 0.06       | 0.92 $\pm$ 0.06         | 0.99 $\pm$ 0.04       | 0.96 $\pm$ 0.04         |
| MITM ARP Spoofing     | 0.74 $\pm$ 0.32      | 0.64 $\pm$ 0.26        | 0.92 $\pm$ 0.18       | 0.80 $\pm$ 0.16         | 0.98 $\pm$ 0.10       | 0.87 $\pm$ 0.11         |
| Mirai-Ackflooding     | 0.71 $\pm$ 0.34      | 0.56 $\pm$ 0.27        | 0.90 $\pm$ 0.21       | 0.64 $\pm$ 0.15         | 0.98 $\pm$ 0.11       | 0.66 $\pm$ 0.06         |
| Mirai-HTTP Flooding   | 0.71 $\pm$ 0.34      | 0.55 $\pm$ 0.27        | 0.90 $\pm$ 0.21       | 0.64 $\pm$ 0.15         | 0.98 $\pm$ 0.11       | 0.66 $\pm$ 0.07         |
| Mirai-Hostbruteforceg | 0.71 $\pm$ 0.25      | 0.61 $\pm$ 0.20        | 0.84 $\pm$ 0.15       | 0.67 $\pm$ 0.10         | 0.91 $\pm$ 0.08       | 0.67 $\pm$ 0.05         |
| Mirai-UDP Flooding    | 0.71 $\pm$ 0.31      | 0.53 $\pm$ 0.30        | 0.94 $\pm$ 0.13       | 0.80 $\pm$ 0.09         | 0.99 $\pm$ 0.06       | 0.84 $\pm$ 0.05         |
| Scan Hostport         | 0.73 $\pm$ 0.13      | 0.68 $\pm$ 0.11        | 0.80 $\pm$ 0.04       | 0.75 $\pm$ 0.02         | 0.83 $\pm$ 0.03       | 0.75 $\pm$ 0.01         |
| Scan Port OS          | 0.76 $\pm$ 0.14      | 0.72 $\pm$ 0.13        | 0.84 $\pm$ 0.04       | 0.79 $\pm$ 0.02         | 0.87 $\pm$ 0.02       | 0.79 $\pm$ 0.01         |
| All Attacks           | 0.73 $\pm$ 0.25      | 0.61 $\pm$ 0.20        | 0.90 $\pm$ 0.13       | 0.75 $\pm$ 0.09         | 0.95 $\pm$ 0.07       | 0.78 $\pm$ 0.05         |


## How To Run
The configurations in these repo are managed by hydra. <br/>
See an example for command line arguments usage [here](https://hydra.cc/docs/0.11/tutorial/simple_cli/).

### Training
Config is located at ```configs/train.yaml```.
```
python train.py dataset=iot_intrusion_dataset training_loop=classification_training_loop model=simple_nn
python train.py dataset=iot_intrusion_dataset training_loop=autoencoder_training_loop model=autoencoder
python train.py dataset=iot_intrusion_dataset training_loop=sklearn_training_loop model=one_class_svm
```

### Evaluation
Config is located at ```configs/eval.yaml```.
```
python eval.py dataset=iot_intrusion_dataset evaluation=zero_shot checkpoint_path=PATH_TO_CKPT_FILE
python eval.py dataset=iot_intrusion_dataset evaluation=few_shot checkpoint_path=PATH_TO_CKPT_FILE
```
