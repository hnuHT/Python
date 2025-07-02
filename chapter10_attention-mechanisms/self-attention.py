import math
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# -------------------------------
# 位置编码模块（正余弦函数实现）
# -------------------------------
class PositionalEncoding(nn.Module):
    """位置编码（基于正余弦函数）"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码张量 shape: (1, max_len, num_hiddens)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

        self.P[:, :, 0::2] = torch.sin(X)   # 偶数位置用 sin 编码
        self.P[:, :, 1::2] = torch.cos(X)   # 奇数位置用 cos 编码

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)  # 添加位置编码
        return self.dropout(X)


# -------------------------------
# 演示自注意力计算
# -------------------------------
def run_self_attention_example():
    num_hiddens, num_heads = 100, 5
    batch_size, num_queries = 2, 4
    valid_lens = torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))

    # 构建多头自注意力模块（Q=K=V）
    attention = d2l.MultiHeadAttention(
        key_size=num_hiddens, query_size=num_hiddens,
        value_size=num_hiddens, num_hiddens=num_hiddens,
        num_heads=num_heads, dropout=0.5
    )
    attention.eval()

    # 自注意力前向传播
    out = attention(X, X, X, valid_lens)
    print("Self-Attention output shape:", out.shape)  # [2, 4, 100]


# -------------------------------
# 可视化位置编码
# -------------------------------
def visualize_positional_encoding():
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, dropout=0)
    pos_encoding.eval()

    # 输入为零向量，仅看位置编码效果
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :num_steps, :]

    d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T,
             xlabel='Row (position)', figsize=(6, 2.5),
             legend=["Col %d" % d for d in torch.arange(6, 10)])
    plt.show()


# -------------------------------
# 热图可视化位置编码矩阵
# -------------------------------
def show_positional_encoding_heatmap():
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, dropout=0)
    P = pos_encoding.P[:, :num_steps, :]  # shape: (1, num_steps, dim)
    P = P[0, :, :].unsqueeze(0).unsqueeze(0)

    d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                      ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
    plt.show()


# -------------------------------
# 主程序入口
# -------------------------------
if __name__ == "__main__":
    run_self_attention_example()
    visualize_positional_encoding()
    show_positional_encoding_heatmap()
