from abc import ABC
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter


class TrafficDataset(ABC, Dataset):
    
    def __init__(self, path, is_train, lbl_is_src_ip,
                 include_malicious_traffic, src_ip_clm_nm,
                 dst_ip_clm_nm, label_clm_nm, flow_id_clms,
                 categorical_clms, all_label_related_clms, overfit_clms,
                 benign_label_in_dataset, test_size=0.2):
        # if lbl_is_src_ip is True, then the label is the source IP address
        # otherwise, the label is the attack subcategory
        
        if include_malicious_traffic:
            assert not is_train, 'Malicious traffic is only included in the test set'
        
        super().__init__()
        self.path = path
        self.is_train = is_train
        self.lbl_is_src_ip = lbl_is_src_ip
        self.include_malicious_traffic = include_malicious_traffic
        self.src_ip_clm_nm = src_ip_clm_nm
        self.dst_ip_clm_nm = dst_ip_clm_nm
        self.label_clm_nm = label_clm_nm
        self.flow_id_clms = flow_id_clms
        self.categorical_clms = categorical_clms
        self.all_label_related_clms = all_label_related_clms
        self.overfit_clms = overfit_clms
        self.benign_label_in_dataset = benign_label_in_dataset
        
        self.test_size = test_size
        
        self.num_classes = None
        self.X = None
        self.y = None
        self.benign_label = None
        self.label_encoder = None
        
        X_benign, X_malicious, y_benign, y_malicious = self._get_X_y()
        if not include_malicious_traffic:
            # if we don't need it, make it None so we don't use it by accident
            X_malicious = None
            y_malicious = None
                
        # if element in y appears only once, delete it
        X_benign, y_benign = self._remove_single_occurences(X_benign, y_benign)
        y_benign, y_malicious = self._label_encode(y_benign, y_malicious)

        X_train_benign, X_test_benign, y_train_benign, y_test_benign = train_test_split(X_benign, y_benign,
                                                                                        test_size=test_size,
                                                                                        random_state=42)
        
        preprocessed_X_train_benign, scaler = self._preprocess(X_train_benign)
        self.clms = X_train_benign.columns
        self._set_preprocessed_X_y(preprocessed_X_train_benign, y_train_benign,
                                   X_test_benign, y_test_benign, scaler,
                                   X_malicious, y_malicious)
        self.row_size = self.X.shape[1]
    
    def _set_preprocessed_X_y(self, preprocessed_X_train_benign, y_train_benign,
                              X_test_benign, y_test_benign, scaler, X_malicious=None, y_malicious=None):
        
        if self.is_train:
            self.X = preprocessed_X_train_benign.values.astype(np.float32)
            self.y = y_train_benign.astype(np.int64) # TODO : should convert?
        else:
            X_test_benign, _ = self._preprocess(X_test_benign, scaler)
            
            if self.include_malicious_traffic:
                X_malicious, _ = self._preprocess(X_malicious, scaler)
                X_test = pd.concat([X_test_benign, X_malicious])
                y_test = np.concatenate([y_test_benign, y_malicious])
            else:
                X_test = X_test_benign
                y_test = y_test_benign
                
            self.X = X_test.values.astype(np.float32)
            self.y = y_test.astype(np.int64)  # TODO : should convert?
            
    def _label_encode(self, y_benign, y_malicious=None):
        if y_malicious is not None:
            y = pd.concat([y_benign, y_malicious])
        else:
            y = y_benign
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        self.num_classes = len(self.label_encoder.classes_)
        
        y_benign = self.label_encoder.transform(y_benign)
        if y_malicious is not None:
            y_malicious = self.label_encoder.transform(y_malicious)
            self.benign_label = self.label_encoder.transform([self.benign_label_in_dataset])[0]
        else:
            y_malicious = None
        
        return y_benign, y_malicious
    
    def _remove_single_occurences(self, X, y):
        cntr = Counter(y)
      
        for key in list(cntr.keys()):
            if cntr[key] == 1:
                X = X[y != key]
                y = y[y != key]
        
        return X, y
    
    def _preprocess(self, X, scaler=None):
        normalized_ips = self._split_ip_address_and_normalize(X[self.dst_ip_clm_nm], 'dst_')
        to_normalize = X.drop([self.dst_ip_clm_nm], axis=1)

        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(to_normalize)
        
        # there's a small amount of infs in the data, so we replace them with 0
        to_normalize = to_normalize.replace([np.inf, -np.inf], 0)
        normalized_X = pd.DataFrame(scaler.transform(to_normalize), columns=to_normalize.columns)
        preprocessed_X = pd.concat([normalized_ips, normalized_X], axis=1)
        
        # Found using training a random forest on the data and observing the feature importances
        preprocessed_X = preprocessed_X.drop(self.overfit_clms, axis=1)
        
        return preprocessed_X, scaler
    
    def _get_X_y(self):
        data = pd.read_csv(self.path)
        # TODO: categorical clms are currently dropped, might want to use them

        data = data.drop(self.flow_id_clms + self.categorical_clms, axis=1)
        X = data.drop(self.all_label_related_clms, axis=1)
        
        if self.lbl_is_src_ip:
            y = data[self.src_ip_clm_nm]
        else:
            y = data[self.label_clm_nm]
        
        benign_mask = data[self.label_clm_nm] == self.benign_label_in_dataset
        X_benign, y_benign = X[benign_mask], y[benign_mask]
        X_malicious, y_malicious = X[~benign_mask], y[~benign_mask]
        return X_benign, X_malicious, y_benign, y_malicious
                
    def _split_ip_address_and_normalize(self, ip_address: pd.Series, prefix: str):
        ip_address_split = ip_address.apply(lambda x: list(map(lambda val: int(val) / 255, x.split('.'))))
        return pd.DataFrame(ip_address_split.tolist(),
                            columns=[prefix + 'ip_1', prefix + 'ip_2', prefix + 'ip_3', prefix + 'ip_4'])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]