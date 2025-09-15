import torch
import torch.nn as nn

from LayerNormal import LayerNorml
from MultiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, num_head, d_model, ffn_hidden, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(num_head, d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = LayerNorml(d_model)

        self.cross_multi_head_attention = MultiHeadAttention(num_head, d_model)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = LayerNorml(d_model)

        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer_norm3 = LayerNorml(d_model)

    #trg_mask确保解码器在预测第 t个 token 时，只能看到第 1到 t-1的 token
    #trg_mask屏蔽源序列中的填充 token（<pad>），防止注意力机制计算无效位置。
    def forward(self, dec_x, enc_x, trg_mask, src_mask):
        #解码器的第一层自注意力要用trg_mask遮盖正确答案，放在解码器看到未来信息作弊
        temp = self.multi_head_attention(q=dec_x, k=dec_x, v=dec_x, mask=trg_mask)
        temp = self.dropout1(temp)
        temp = self.layer_norm1(temp + enc_x)

        #第二个注意力层是交叉注意力层，
        temp_temp = temp
        if enc_x is not None:
            temp_temp = self.cross_multi_head_attention(q=temp, k=enc_x, v=enc_x, mask=src_mask)
            temp_temp = self.dropout2(temp_temp)
            temp_temp = self.layer_norm2(temp_temp + temp)

        output = self.ffn(temp_temp)
        output = self.dropout3(output)
        output = self.layer_norm3(output + temp_temp)
        return output