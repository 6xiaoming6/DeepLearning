import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        #第一个参数num_embeddings是词汇表的长度，第二个参数embedding_dim是每个词向量的长度
        #参数padding_index是为了在句子长度不一致时填充句子使其都保持长度一致而使用的。
        #整个embedding矩阵就是一个vocab_size x d_model的矩阵，我们输入单词的索引，用这个索引从这个矩阵中查表，得到对应的词向量。
        #padding_index所对应的embedding矩阵行参数会固定为0，这样用于padding_index填充句子时不会让填充值影响输出结果
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
        self.scale = torch.sqrt(torch.tensor(d_model))  # 缩放因子
        self.device = device

    def forward(self, x):
        output = super().forward(x) * self.scale
        return output.to(self.device)


class PositionEmbedding(nn.Module):
    #max_len是句子的最大长度
    def __init__(self, d_model, max_len, device):
        super(PositionEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.require_grad = False

        pos = torch.arange(0, max_len, device=device).float()
        # 后面添加一个纬度应用广播机制，否则会报错
        pos = pos.unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 等号右边的一堆式子计算出的形状为(max_len, d_model/2)
        # 下面两个式子分别计算了每个位置通道数为奇数和偶数的位置编码，最终encoding矩阵里面存到为max_len个词向量的位置编码
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i/d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i/d_model)))

    def forward(self, x):
        #拿到该批量样本的句子长度
        batch_size, seq_len = x.size()
        #前seq_len个词的位置编码从矩阵中拿出
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_out_rate, device):
        super(TransformerEmbedding, self).__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(drop_out_rate)

    def forward(self, x):
        t = self.token_embedding(x) #输出形状为(batch_size, seq_len, d_model)
        p = self.position_embedding(x) #输出形状为(seq_len, d_model)

        return self.drop_out(t + p) #广播后二者相加，形状为(batch_size, seq_len, d_model)，再经过一次drop_out层


if __name__ == '__main__':
    vocba_size = 100
    d_model = 32
    max_len = 10
    em = TransformerEmbedding(vocba_size, d_model, max_len)

    x = torch.tensor([
        [0,2,3,4,5],
        [6,7,8,9,1],
    ])
    output = em(x)
    print(output.shape)
