from datasets.traffic_dataset import TrafficDataset


class MqttDataset(TrafficDataset):
    # dataset of  https://ieee-dataport.org/open-access/mqtt-iot-ids2020-mqtt-internet-things-intrusion-detection-dataset
    
    def __init__(self, path, is_train, lbl_is_src_ip, include_malicious_traffic, test_size=0.2):        
        super().__init__(path, is_train, lbl_is_src_ip,include_malicious_traffic, 
                        src_ip_clm_nm='src_ip',
                        dst_ip_clm_nm='dst_ip', 
                        label_clm_nm='category',
                        flow_id_clms=['timestamp'],
                        categorical_clms=['protocol', 'mqtt_messagetype'],
                        all_label_related_clms=['category', 'src_ip'],
                        overfit_clms= [],
                        benign_label_in_dataset='Normal',
                        test_size=test_size)
