import torch
from torch.utils.data import Dataset
import numpy as np

MAX_PACKETS = 50
MAX_BYTES = 256

class HierarchicalFlowDataset(Dataset):
    def __init__(self, flows, max_packets=MAX_PACKETS, max_bytes=MAX_BYTES):
        self.max_packets = max_packets
        self.max_bytes = max_bytes
        self.flows = flows
        
    def __len__(self): return len(self.flows)
    
    def __getitem__(self, idx):
        flow = self.flows[idx]
        raw_label = flow['label'].strip().upper()
        label = 0 if raw_label == 'BENIGN' else 1
        packets = flow['packets']
        seq_len = min(len(packets), self.max_packets)
        
        flow_bytes = np.full((self.max_packets, self.max_bytes), 256, dtype=np.int64)
        flow_protos = np.zeros(self.max_packets, dtype=np.int64)
        flow_lens = np.zeros(self.max_packets, dtype=np.float32)
        flow_iats = np.zeros(self.max_packets, dtype=np.float32)
        flow_dirs = np.zeros(self.max_packets, dtype=np.int64)
        attention_mask = np.ones(self.max_packets, dtype=np.bool_) 

        for i in range(seq_len):
            pkt = packets[i]
            flow_bytes[i, :min(len(pkt['byte_vector']), self.max_bytes)] = pkt['byte_vector'][:self.max_bytes] 
            flow_protos[i] = pkt.get('protocol', 0)
            flow_lens[i] = pkt.get('original_len', 0) / 1500.0
            flow_iats[i] = np.log1p(pkt.get('iat', 0.0))
            flow_dirs[i] = 1 if pkt.get('direction', 0) == 1 else 0
            attention_mask[i] = False 
        
        return {
            'byte_vectors': torch.tensor(flow_bytes, dtype=torch.long), 
            'protocols': torch.tensor(flow_protos, dtype=torch.long),
            'orig_lens': torch.tensor(flow_lens, dtype=torch.float32), 
            'iats': torch.tensor(flow_iats, dtype=torch.float32),
            'directions': torch.tensor(flow_dirs, dtype=torch.long), 
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.long),
        }

