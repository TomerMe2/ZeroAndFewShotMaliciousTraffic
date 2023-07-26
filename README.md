# Learn to Detect Attacks Without Seeing Attacks

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

| Attack                | 1-shot @0.2 | 1-shot @0.25 | 10-shot @0.2| 10-shot @0.25 | 30-shot @0.2 | 30-shot @0.25 |
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
python eval.py dataset=iot_intrusion_dataset evaluation=zero_shot
python eval.py dataset=iot_intrusion_dataset evaluation=few_shot
```
