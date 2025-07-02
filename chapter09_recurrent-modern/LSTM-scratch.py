import torch
from torch import nn
from d2l import torch as d2l

# ==========================
# 1. 加载时间机器数据集
# ==========================
batch_size, num_steps = 32, 35  # 每批处理32个样本，每个样本长度为35个字符
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# ============================================
# 2. 初始化模型参数（从零开始构建LSTM参数）
# ============================================
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    # 初始化一个小标准差的正态分布权重
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 返回三元组：(W_x, W_h, b)
    def three():
        return (
            normal((num_inputs, num_hiddens)),   # 输入到门的权重
            normal((num_hiddens, num_hiddens)),  # 隐状态到门的权重
            torch.zeros(num_hiddens, device=device)  # 偏置
        )

    # 输入门参数
    W_xi, W_hi, b_i = three()
    # 遗忘门参数
    W_xf, W_hf, b_f = three()
    # 输出门参数
    W_xo, W_ho, b_o = three()
    # 候选记忆元参数
    W_xc, W_hc, b_c = three()
    # 输出层参数（将隐状态映射到词表大小的输出）
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 所有参数收集在一起
    params = [
        W_xi, W_hi, b_i,
        W_xf, W_hf, b_f,
        W_xo, W_ho, b_o,
        W_xc, W_hc, b_c,
        W_hq, b_q
    ]

    # 所有参数需要参与梯度计算
    for param in params:
        param.requires_grad_(True)

    return params

# ==================================================
# 3. 初始化LSTM状态 (返回隐状态H 和 记忆元C)
# ==================================================
def init_lstm_state(batch_size, num_hiddens, device):
    return (
        torch.zeros((batch_size, num_hiddens), device=device),  # H: 隐状态
        torch.zeros((batch_size, num_hiddens), device=device)   # C: 记忆元
    )

# ================================
# 4. 定义LSTM模型的前向传播逻辑
# ================================
def lstm(inputs, state, params):
    (
        W_xi, W_hi, b_i,
        W_xf, W_hf, b_f,
        W_xo, W_ho, b_o,
        W_xc, W_hc, b_c,
        W_hq, b_q
    ) = params

    H, C = state  # 初始化时为全0
    outputs = []

    for X in inputs:  # 遍历时间步，每个X形状：(batch_size, vocab_size)
        # 输入门、遗忘门、输出门
        I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)  # 输入门
        F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)  # 遗忘门
        O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)  # 输出门

        # 候选记忆元
        C_tilda = torch.tanh(X @ W_xc + H @ W_hc + b_c)

        # 更新记忆元
        C = F * C + I * C_tilda

        # 计算新的隐状态
        H = O * torch.tanh(C)

        # 计算输出
        Y = H @ W_hq + b_q
        outputs.append(Y)

    # 按时间步拼接输出，形状：(时间步数 * batch_size, vocab_size)
    return torch.cat(outputs, dim=0), (H, C)

# ==========================
# 5. 训练模型
# ==========================
vocab_size, num_hiddens = len(vocab), 256
device = d2l.try_gpu()
num_epochs, lr = 500, 1

# 用我们从零实现的函数构建模型（注意不是PyTorch的nn.LSTM）
model = d2l.RNNModelScratch(
    vocab_size, num_hiddens, device,
    get_lstm_params, init_lstm_state, lstm
)

# 开始训练并进行预测
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
