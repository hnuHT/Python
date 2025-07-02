import torch
from d2l import torch as d2l

# -----------------------------
# 1. 初始化 Adadelta 所需状态变量
# -----------------------------
# 每个参数都需要两个状态变量：
# - s：平方梯度的指数加权平均
# - delta：参数更新值的平方的指数加权平均
def init_adadelta_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))       # 权重的平方梯度累积
    delta_w = torch.zeros((feature_dim, 1))   # 权重更新的平方累积
    s_b = torch.zeros(1)                      # 偏置的平方梯度累积
    delta_b = torch.zeros(1)                  # 偏置更新的平方累积
    return ((s_w, delta_w), (s_b, delta_b))


# -----------------------------
# 2. Adadelta 优化器的更新函数
# -----------------------------
def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5  # rho 是衰减率，eps 是稳定常数
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # 更新梯度平方的加权平均
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            # 计算缩放后的更新量（即学习率被动态调整）
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            # 应用更新
            p[:] -= g
            # 更新参数变化值的平方平均
            delta[:] = rho * delta + (1 - rho) * g * g
        # 梯度清零
        p.grad.data.zero_()


# -----------------------------
# 3. 准备数据并进行训练
# -----------------------------
# 加载线性回归数据集
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)

# 使用 Adadelta 算法进行训练
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim)


# from torch.optim import Adadelta
# trainer = Adadelta
# d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)

import matplotlib.pyplot as plt
plt.show()
