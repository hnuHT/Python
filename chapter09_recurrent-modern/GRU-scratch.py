import torch
from torch import nn
from d2l import torch as d2l

# 1. 读取数据
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 2. 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (
            normal((num_inputs, num_hiddens)),  # W_x*
            normal((num_hiddens, num_hiddens)), # W_h*
            torch.zeros(num_hiddens, device=device)  # b_*
        )

    # 更新门参数
    W_xz, W_hz, b_z = three()
    # 重置门参数
    W_xr, W_hr, b_r = three()
    # 候选隐状态参数
    W_xh, W_hh, b_h = three()
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xz, W_hz, b_z,
              W_xr, W_hr, b_r,
              W_xh, W_hh, b_h,
              W_hq, b_q]

    for param in params:
        param.requires_grad_(True)

    return params

# 3. 初始化隐藏状态
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 4. 定义GRU模型
def gru(inputs, state, params):
    W_xz, W_hz, b_z, \
    W_xr, W_hr, b_r, \
    W_xh, W_hh, b_h, \
    W_hq, b_q = params

    H, = state
    outputs = []

    for X in inputs:
        Z = torch.sigmoid(X @ W_xz + H @ W_hz + b_z)
        R = torch.sigmoid(X @ W_xr + H @ W_hr + b_r)
        H_tilda = torch.tanh(X @ W_xh + (R * H) @ W_hh + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H,)

# 5. 模型训练与预测
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1

model = d2l.RNNModelScratch(
    vocab_size, num_hiddens, device,
    get_params, init_gru_state, gru
)

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
