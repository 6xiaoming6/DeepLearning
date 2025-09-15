import torch
import torch.nn as nn
from ScaleDotProductAttention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.attention = ScaleDotProductAttention()

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        #将多个头输出的结果concat后再经过一个线性层
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) #(batch_size, seq_len, d_model)
        q, k, v = self.split_heads(q), self.split_heads(v), self.split_heads(v) #(batch_size, num_head, seq_len, d_tensor)

        output, attention_score = self.attention(q, k, v, mask=mask) #(batch_size, num_head, seq_len, d_tensor)
        #把多个头concat一起
        output = self.concat_heads(output)
        #contact后再经过一层线性层投影
        output = self.w_concat(output)
        return output

    #将d_model的qkv拆分为num_head个qkv
    def split_heads(self, tensor):
        batch_size, seq_len, d_model = tensor.size()

        d_tensor = d_model // self.num_head
        # 拆分后转换为 (batch_size, num_head, seq_len, d_tensor)，便于在多头注意力中使用矩阵运算，而不是循环计算num_head次
        tensor = tensor.view(batch_size, seq_len, self.num_head, d_tensor).transpose(1, 2)

        return tensor

    #将之前拆分的多头合并到一起
    def concat_heads(self, tensor):
        batch_size, num_head, seq_len, d_tensor = tensor.size()

        #张量执行transpose()后需要调用contiguous()方法才能调用view()方法，否则会报错
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, num_head * d_tensor)

        return tensor
