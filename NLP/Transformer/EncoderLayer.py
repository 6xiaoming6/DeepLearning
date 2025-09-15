import torch
import torch.nn as nn

from LayerNormal import LayerNorml
from MultiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, num_head, d_model, ffn_hidden, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(num_head, d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = LayerNorml(d_model)

        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = LayerNorml(d_model)

    def forward(self, x, src_mask):
        temp = self.multi_head_attention(q=x, k=x, v=x, mask=src_mask)
        temp = self.dropout1(temp)
        temp = self.layer_norm1(x + temp)

        output = self.ffn(temp)
        output = self.dropout2(output)
        output = self.layer_norm2(temp + output)

        return output