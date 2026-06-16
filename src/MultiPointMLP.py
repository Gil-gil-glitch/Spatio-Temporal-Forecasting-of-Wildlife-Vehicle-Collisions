#
## MultiPointMLP.py
#
# Multi-Point Multilayer Perceptron for Direct Mapping of Historical Spatial Segment Sequences to Future Risk Estimates
#

import torch
import torch.nn as nn

class MultiPointMLP(nn.Module):
    def __init__(self, lookback, num_segments):
        super(MultiPointMLP, self).__init__()
        self.flatten_dim = lookback * num_segments
        self.network = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_segments) # Mapping out to direct 96 spatial bins
        )
    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten temporal tracking dimensions
        return self.network(x)