#
## StackedLSTM.py
#
# This model implements a multi-layer deep recurrent architecture designed to learn hierarchical 
# temporal abstractions within the sequence arrays. It consists of multiple LSTM layers stacked 
#  on top of each other, allowing the model to capture more complex temporal patterns in the data. 
# The final output is obtained from the last hidden state of the top LSTM layer, which is then 
# passed through a fully connected layer to produce the risk estimates for each segment at the 
# forecast horizon.
#

import torch
import torch.nn as nn



class StackedLSTM(nn.Module):
    """
    Multi-layer deep recurrent architecture designed to learn hierarchical 
    temporal abstractions within the sequence arrays.
    """
    def __init__(self, input_dim=96, hidden_dim=128, output_dim=96, num_layers=3, dropout=0.2):
        super(StackedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x) # Input shape: (Batch, Seq_Len, Input_Dim)
        out = self.fc(lstm_out[:, -1, :])
        return out
