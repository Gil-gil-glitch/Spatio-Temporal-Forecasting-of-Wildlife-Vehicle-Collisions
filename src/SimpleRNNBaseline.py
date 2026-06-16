#
## Simple RNN Baseline for sequence modeling
#
# A straightforward RNN architecture that processes historical spatial segment sequences to predict future risk estimates without incorporating exogenous environmental or temporal signals. This serves as a baseline for evaluating the performance of more complex architectures like the Dual-Branch LSTM.
#

import torch
import torch.nn as nn

class SimpleRNNBaseline(nn.Module):
    def __init__(self, num_segments, hidden_dim=128):
        super(SimpleRNNBaseline, self).__init__()
        self.rnn = nn.RNN(input_size=num_segments, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_segments)
        
    def forward(self, x):
        # x shape: (Batch, Sequence, Segments)
        out, _ = self.rnn(x)
        # Pulling out final hidden step output state vector
        out = self.fc(out[:, -1, :])
        return out