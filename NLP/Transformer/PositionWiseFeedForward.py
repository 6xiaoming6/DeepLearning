import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, num_hidden, dropout_rate = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.d_model = d_model

        self.layer = nn.Sequential(
            nn.Linear(d_model, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_hidden, d_model),
        )

    def forward(self, x):
        return self.layer(x)