import torch
import torch.nn as nn

class LayerNorml(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorml, self).__init__()

        self.gamma = nn.parameter.Parameter(torch.ones(d_model))
        self.beta = nn.parameter.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)

        output = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * output + self.beta
