#
## BidirectionalLSTM.py
#
# This model implements a bidirectional LSTM architecture to capture both forward and backward temporal dependencies. 
# It processes sequences of segment-level crash tensors in both directions, allowing the model to learn from past and 
# future contexts simultaneously. The final output is obtained from the last hidden state of the top LSTM layer, which 
# is then passed through a fully connected layer to produce the risk estimates for each segment at the forecast horizon.
#
#

import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """
    Dual-directional recurrent layer that tracks both forward historical updates 
    and reverse contextual indicators across the tensor's lookback timeline.
    """
    def __init__(self, input_dim=96, hidden_dim=128, output_dim=96, num_layers=1):
        super(BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        # Bidirectional tracking doubles the latent space dimension (hidden * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out