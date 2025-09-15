import torch.nn as nn
import torch.nn.functional as F
import torch




if __name__ == '__main__':
    x = torch.tensor([
    [100,  50,  30,  -1e9],  # 第1行：屏蔽第4列
    [ 80,  70,  40,  -1e9],  # 第2行：屏蔽第4列
    [ 60,  40,  90,  -1e9],  # 第3行：屏蔽第4列
    [-1e9, -1e9, -1e9, -1e9] # 第4行：全部屏蔽
])
    print(x)
