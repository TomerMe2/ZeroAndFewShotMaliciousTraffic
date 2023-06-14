from typing import Any
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter


class IotIntrusionDataset(Dataset):
    # dataset of https://sites.google.com/view/iot-network-intrusion-dataset/home

    def __init__(self, path, is_train, lbl_is_src_ip, include_malicious_traffic, test_size=0.2):
        # if lbl_is_src_ip is True, then the label is the source IP address
        # otherwise, the label is the attack subcategory
        
        if include_malicious_traffic:
            assert not is_train, 'Malicious traffic is only included in the test set'
        
        super().__init__()
        self.path = path
        
        data = pd.read_csv(path)
        # TODO: might want to drop Dst_port as well?
        # TODO: Protocol might be nice, but it should be one hot encoded
        data = data.drop(['Flow_ID', 'Timestamp', 'Protocol'] , axis=1)
        X = data.drop(['Label', 'Cat', 'Sub_Cat', 'Src_IP'], axis=1)
        
        if lbl_is_src_ip:
            y = data['Src_IP']
        else:
            y = data['Sub_Cat']
        
        if not include_malicious_traffic:
            X = X[data['Sub_Cat'] == 'Normal']
            y = y[data['Sub_Cat'] == 'Normal']
        
        
        # TODO: if include_malicious_traffic, we need to add malicious traffic ONLY to the test
        
        # if element in y appears only once, delete it
        cntr = Counter(y)
        for key in list(cntr.keys()):
            if cntr[key] == 1:
                X = X[y != key]
                y = y[y != key]
        
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.num_classes = len(le.classes_)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        X_train, scaler = self._preprocess(X_train)
        self.clms = X_train.columns
        
        if is_train:
            self.X = X_train.values.astype(np.float32)
            self.y = y_train
        else:
            X_test, _ = self._preprocess(X_test, scaler)
            self.X = X_test.values.astype(np.float32)
            self.y = y_test
        
        self.row_size = self.X.shape[1]
    
    def _preprocess(self, X, scaler=None):
        normalized_ips = self._split_ip_address_and_normalize(X['Dst_IP'], 'dst_')
        to_normalize = X.drop(['Dst_IP'], axis=1)

        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(to_normalize)
        
        normalized_X = pd.DataFrame(scaler.transform(to_normalize), columns=to_normalize.columns)
        return pd.concat([normalized_ips, normalized_X], axis=1), scaler
                
    def _split_ip_address_and_normalize(self, ip_address: pd.Series, prefix: str):
        ip_address_split = ip_address.apply(lambda x: list(map(lambda val: int(val) / 255, x.split('.'))))
        return pd.DataFrame(ip_address_split.tolist(), columns=[prefix + 'ip_1', prefix + 'ip_2', prefix + 'ip_3', prefix + 'ip_4'])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
