import torch
import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_padding_idx, trg_padding_idx, trg_sos_idx, enc_vocab_size, dec_vocab_size,
                 d_model, num_head, max_len, ffn_hidden, num_layer, dropout_rate, device):
        super(Transformer, self).__init__()

        self.src_padding_idx = src_padding_idx
        self.trg_padding_idx = trg_padding_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(enc_vocab_size, max_len, num_head, d_model,ffn_hidden, num_layer, dropout_rate, device)
        self.decoder = Decoder(dec_vocab_size, max_len, num_head, d_model, ffn_hidden, num_layer, dropout_rate, device)

    #此时传进来的数据为(batch_size, seq_len)，嵌入层在编码器和解码器内部
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        z = self.encoder(src, src_mask)
        output = self.decoder(trg, z, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        #这里增加纬度是为了后面替换score矩阵时广播使用，否则不能得到正确的结果
        src_mask = (src != self.src_padding_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        #解码器输入的是正确答案的句子，也有可能输入padding，所有也要一个mask去除掉padding的影响
        trg_pad_mask = (trg != self.trg_padding_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        #tril()函数取矩阵的下三角部分（包含对角线），上三角部分置 0。得到下三角的mask矩阵
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        #最终的解码器矩阵就是去除padding的pad_mask和遮盖未来信息的trg_mask取与操作
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask


if __name__ == '__main__':
    pass




