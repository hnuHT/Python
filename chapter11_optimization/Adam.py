import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# ---------------------------
# 1. 初始化Adam和Yogi的状态变量
# 每个参数需要两个状态变量：一阶动量v，二阶动量s
# ---------------------------
def init_adam_states(feature_dim):
    v_w = torch.zeros((feature_dim, 1))  # 权重的一阶动量
    s_w = torch.zeros((feature_dim, 1))  # 权重的二阶动量
    v_b = torch.zeros(1)                 # 偏置的一阶动量
    s_b = torch.zeros(1)                 # 偏置的二阶动量
    return ((v_w, s_w), (v_b, s_b))


# ---------------------------
# 2. Adam优化器的实现（带偏差修正）
# ---------------------------
def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            # 更新一阶动量
            v[:] = beta1 * v + (1 - beta1) * p.grad
            # 更新二阶动量（平方梯度的指数平均）
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            # 偏差修正（防止前期值偏小）
            v_corr = v / (1 - beta1 ** hyperparams['t'])
            s_corr = s / (1 - beta2 ** hyperparams['t'])
            # 参数更新
            p[:] -= hyperparams['lr'] * v_corr / (torch.sqrt(s_corr) + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1


# ---------------------------
# 3. Yogi优化器的实现
# ---------------------------
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            # 一阶动量更新
            v[:] = beta1 * v + (1 - beta1) * p.grad
            # Yogi特有的二阶动量更新方式（符号控制，防止s爆炸）
            s[:] = s + (1 - beta2) * torch.sign(torch.square(p.grad) - s) * torch.square(p.grad)
            # 偏差修正
            v_corr = v / (1 - beta1 ** hyperparams['t'])
            s_corr = s / (1 - beta2 ** hyperparams['t'])
            # 参数更新
            p[:] -= hyperparams['lr'] * v_corr / (torch.sqrt(s_corr) + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1


# ---------------------------
# 4. 使用 Adam/Yogi 训练模型
# ---------------------------
# 加载小批量数据集（用于线性回归）
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)

# 训练模型：可选择 adam 或 yogi
print("使用 Adam 优化器训练模型")
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim)

# print("使用 Yogi 优化器训练模型")
# d2l.train_ch11(yogi, init_adam_states(feature_dim),
#                {'lr': 0.01, 't': 1}, data_iter, feature_dim)


# ---------------------------
# 5. 使用PyTorch内置Adam优化器
# ---------------------------
# from torch.optim import Adam
# trainer = Adam
# d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)


# ---------------------------
# 6. 显示训练过程图像
# ---------------------------
plt.show()
