import torch
from torch import nn
import math


# 缩放点积注意力机制
class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)  # (B*H, Q, K)

        if valid_lens is not None:
            # valid_lens: (B * H,) 一维张量
            max_len = scores.shape[-1]  # K
            # 构造 mask: (B*H, K)
            mask = torch.arange(max_len, device=queries.device)[None, :] >= valid_lens[:, None]
            # 扩展为 (B*H, 1, K)，以广播到 scores: (B*H, Q, K)
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(attention_weights), values)


# 为多头注意力准备的输入转置函数
def transpose_qkv(X, num_heads):
    # 输入 X: (batch_size, seq_len, num_hiddens)
    batch_size, seq_len, num_hiddens = X.shape
    assert num_hiddens % num_heads == 0, "num_hiddens 必须能整除 num_heads"
    new_hid = num_hiddens // num_heads
    # 输出: (batch_size * num_heads, seq_len, num_hiddens / num_heads)
    return X.reshape(batch_size, seq_len, num_heads, new_hid).permute(0, 2, 1, 3).reshape(-1, seq_len, new_hid)


# 多头注意力输出的逆转置
def transpose_output(X, num_heads):
    # 输入 X: (batch_size * num_heads, seq_len, num_hiddens / num_heads)
    batch_head, seq_len, hid = X.shape
    batch_size = batch_head // num_heads
    return X.reshape(batch_size, num_heads, seq_len, hid).permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)


# 多头注意力机制类
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, num_heads, dropout=0.1, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        # 分别对 query、key、value 做线性变换（共享同一组变换）
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        # 线性变换并重排维度
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # valid_lens: (batch_size,) => (batch_size * num_heads,)
            if valid_lens.dim() == 1:
                valid_lens = valid_lens.repeat_interleave(self.num_heads, dim=0)
            else:
                valid_lens = valid_lens.repeat_interleave(self.num_heads, dim=0)

        # 注意力输出: (batch_size * num_heads, seq_len_q, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # 还原多头输出
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


# ===========================
# 测试多头注意力模块
# ===========================
if __name__ == "__main__":
    batch_size, num_queries = 2, 4
    num_kv, num_hiddens, num_heads = 6, 100, 5

    # 模拟输入
    X = torch.ones((batch_size, num_queries, num_hiddens))  # queries
    Y = torch.ones((batch_size, num_kv, num_hiddens))       # keys/values
    valid_lens = torch.tensor([3, 2])                       # 有效键值对数量

    # 实例化多头注意力
    mha = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                             num_hiddens, num_heads, dropout=0.1)
    mha.eval()

    out = mha(X, Y, Y, valid_lens)
    print("Multi-head attention output shape:", out.shape)  # 应为 (2, 4, 100)
