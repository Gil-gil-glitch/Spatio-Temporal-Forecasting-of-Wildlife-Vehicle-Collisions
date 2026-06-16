#
## WVCMultivariateFusionDataset.py
#
#  Custom PyTorch Dataset for Multivariate Fusion of Spatial Segment Sequences and Exogenous Environmental Signals in Wildlife-Vehicle Collision Risk Estimation
#
#
#

import torch
from torch.utils.data import Dataset, DataLoader


class WVCMultivariateFusionDataset(Dataset):
    """
    Custom PyTorch Dataset wrapper mapping synchronized multi-input historical tensors
    and exogenous environmental sequences to their target spatial risk maps.
    """
    def __init__(self, X_hist, X_eco, y):
        self.X_hist = torch.tensor(X_hist, dtype=torch.float32)
        self.X_eco = torch.tensor(X_eco, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X_hist[idx], self.X_eco[idx], self.y[idx]
