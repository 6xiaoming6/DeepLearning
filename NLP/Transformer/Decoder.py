import torch.nn as nn

from NLP.Transformer.DecoderLayer import DecoderLayer
from Embedding import  TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, num_head, d_model, ffn_hidden, num_layer, dropout_rate, device):
        super(Decoder, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout_rate, device)

        self.decoder_layers = nn.ModuleList()
        for i in range(num_layer):
            self.decoder_layers.append(DecoderLayer(num_head, d_model, ffn_hidden, dropout_rate))

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_x, trg_mask, src_mask):
        x = self.embedding(x)

        for layer in self.decoder_layers:
            x = layer(x, enc_x, trg_mask, src_mask)
        return self.linear(x)
