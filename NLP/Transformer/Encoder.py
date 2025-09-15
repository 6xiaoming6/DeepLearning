import torch
import torch.nn as nn

from EncoderLayer import EncoderLayer
from Embedding import  TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, num_head, d_model, ffn_hidden, num_layer, dropout_rate, device):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout_rate, device)

        self.encoder_layers = nn.ModuleList()
        for i in range(num_layer):
            self.encoder_layers.append(EncoderLayer(num_head, d_model, ffn_hidden, dropout_rate))

    def forward(self, x, src_mask):
        x = self.embedding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x


