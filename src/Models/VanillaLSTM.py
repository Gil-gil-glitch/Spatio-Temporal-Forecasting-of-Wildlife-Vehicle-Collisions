#
## VanillaLSTM.py
#
# This model implements a standard single-layer LSTM architecture to capture temporal dependencies 
# in network-wide crash data. It processes sequences of segment-level crash tensors and outputs 
# the risk estimates for each segment at the forecast horizon.
#
#

import torch
import torch.nn as nn

class VanillaLSTM(nn.Module):
    """
    Standard single-layer recurrent gate design to map long-range temporal 
    dependencies over network-wide crash tensors.
    """
    def __init__(self, input_dim=96, hidden_dim=128, output_dim=96, num_layers=1):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x) # Input shape: (Batch, Seq_Len, Input_Dim)
        out = self.fc(lstm_out[:, -1, :])
        return out



