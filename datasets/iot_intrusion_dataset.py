from datasets.traffic_dataset import TrafficDataset


class IotIntrusionDataset(TrafficDataset):
    # dataset of https://sites.google.com/view/iot-network-intrusion-dataset/home

    def __init__(self, path, is_train, lbl_is_src_ip, include_malicious_traffic, test_size=0.2):        
        super().__init__(path, is_train, lbl_is_src_ip,include_malicious_traffic, 
                        src_ip_clm_nm='Src_IP',
                        dst_ip_clm_nm='Dst_IP', 
                        label_clm_nm='Sub_Cat',
                        flow_id_clms=['Flow_ID', 'Timestamp'],
                        categorical_clms=['Protocol'],
                        all_label_related_clms=['Label', 'Cat', 'Sub_Cat', 'Src_IP'],
                        overfit_clms=['dst_ip_4', 'Src_Port', 'Dst_Port'],
                        benign_label_in_dataset='Normal',
                        test_size=test_size)