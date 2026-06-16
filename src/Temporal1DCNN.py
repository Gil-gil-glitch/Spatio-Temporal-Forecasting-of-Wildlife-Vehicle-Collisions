#
## Temporal1DCNN.py
#
#  Temporal 1D Convolutional Neural Network for Localized Temporal Pattern Extraction in Spatial Segment Sequences
#
#
#

class Temporal1DCNN(nn.Module):
    def __init__(self, lookback, num_segments):
        super(Temporal1DCNN, self).__init__()
        # Treating segments as channels to discover parallel localized temporal kernels
        self.conv1 = nn.Conv1d(in_channels=num_segments, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_segments)
        )
    def forward(self, x):
        # Permute input to match PyTorch Conv1d format: (Batch, Channels/Segments, Sequence_Length)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)