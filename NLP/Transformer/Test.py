import torch
import torch.nn as nn
from Transformer import Transformer


# 假设词汇表（简化版）
vocab = {
    "<pad>": 0, "<s>": 1, "</s>": 2,
    "I": 3, "love": 4, "you": 5,
    "我": 6, "爱": 7, "你": 8
}

# 反向词汇表
idx2word = {v: k for k, v in vocab.items()}

# 测试数据（英译中任务）
src_sentences = ["I love you", "I love"]  # 英语（源语言）
trg_sentences = ["我 爱 你", "我 爱"]      # 中文（目标语言）

# 转换为token ID序列
def sentence_to_ids(sentence, vocab):
    return [vocab[word] for word in sentence.split()]

# 填充批次数据到相同长度
def pad_sequences(sequences, max_len, pad_idx):
    padded = torch.full((len(sequences), max_len), pad_idx)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = torch.tensor(seq)
    return padded

# 转换为张量
src_data = [sentence_to_ids(s, vocab) for s in src_sentences]
trg_data = [sentence_to_ids(s, vocab) for s in trg_sentences]

# 填充到最大长度
src_max_len = max(len(s) for s in src_data)
trg_max_len = max(len(s) for s in trg_data)

src_tensor = pad_sequences(src_data, src_max_len, vocab["<pad>"])
trg_tensor = pad_sequences(trg_data, trg_max_len, vocab["<pad>"])

print("源语言数据（英语）:")
print(src_tensor)
print([idx2word[idx.item()] for idx in src_tensor[0]])

print("\n目标语言数据（中文）:")
print(trg_tensor)
print([idx2word[idx.item()] for idx in trg_tensor[0]])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    src_padding_idx=vocab["<pad>"],
    trg_padding_idx=vocab["<pad>"],
    trg_sos_idx=vocab["<s>"],
    enc_vocab_size=len(vocab),
    dec_vocab_size=len(vocab),
    d_model=512,
    num_head=8,
    max_len=100,
    ffn_hidden=2048,
    num_layer=6,
    dropout_rate=0.1,
    device=device
).to(device)

print("\n模型构建成功")


# 将数据移动到设备
src = src_tensor.to(device)
trg = trg_tensor.to(device)

# 前向传播
output = model(src, trg)

print("\n输出张量形状:", output.shape)  # 应为 (batch_size, trg_len, dec_vocab_size)
print("输出示例（第一个样本的最后一个位置）:")
print(output[0, -1, :5])  # 查看第一个样本最后一个位置的预测分布（前5个值）
