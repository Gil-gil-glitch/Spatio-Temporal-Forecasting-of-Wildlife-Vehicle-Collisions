#
## DualBranchFunctionLSTM.py
#
#
#  Dual-Branch LSTM Architecture for Fusing Spatial Segment Sequences with Exogenous Environmental and Temporal Signals
#
#

import torch
import torch.nn as nn

class DualBranchFusionLSTM(nn.Module):
    def __init__(self, num_segments, eco_features, horizon, hidden_dim=128):
        super(DualBranchFusionLSTM, self).__init__()
        self.horizon = horizon
        self.segments = num_segments
        
        # Branch A: Deep Autoregressive Spatial Segment Sequence Encoder
        self.lstm_hist = nn.LSTM(
            input_size=num_segments, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2
        )
        
        # Branch B: Exogenous Environmental (Shapefile) & Temporal Signal Encoder
        self.lstm_eco = nn.LSTM(
            input_size=eco_features, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2
        )
        
        # Multi-Input Fusion and Fully-Connected Regression Layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, horizon * num_segments)  # Projects risk estimates to flat matrix vector (3 * 96)
        )
        
    def forward(self, x_hist, x_eco):
        # Extract the terminal hidden sequence states from Branch A and Branch B
        _, (h_hist, _) = self.lstm_hist(x_hist)
        _, (h_eco, _) = self.lstm_eco(x_eco)
        
        # Pull hidden weights from the top layer of the stacked LSTMs and concatenate
        fused_vector = torch.cat((h_hist[-1], h_eco[-1]), dim=1)
        
        # Map through final regression stack and expand back out to (Batch, 3 Days, 96 Segments)
        risk_outputs = self.fusion_layer(fused_vector)
        return risk_outputs.view(-1, self.horizon, self.segments)