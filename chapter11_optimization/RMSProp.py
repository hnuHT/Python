import math
import torch
from d2l import torch as d2l

# ---------- 定义优化函数使用的二次函数及其可视化 ----------
# 目标函数：f(x1, x2) = 0.1 * x1^2 + 2 * x2^2
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

# RMSProp 在二维示例下的优化轨迹模拟
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # 计算梯度和加上稳定项 epsilon
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2  # s1 是梯度平方的移动平均
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1  # 更新 x1
    x2 -= eta / math.sqrt(s2 + eps) * g2  # 更新 x2
    return x1, x2, s1, s2

# 设置初始学习率和 gamma（衰减因子）
eta, gamma = 0.4, 0.9
# 可视化 RMSProp 在二维空间的优化轨迹
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))

# ---------- RMSProp ----------
# 初始化 RMSProp 的状态变量 s（与权重形状相同）
def init_rmsprop_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))  # 权重的平方梯度累积值
    s_b = torch.zeros(1)  # 偏置项的平方梯度累积值
    return (s_w, s_b)

# RMSProp 参数更新函数
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)  # 更新平方梯度平均值
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)  # 更新参数
        p.grad.data.zero_()  # 清空梯度

# 获取数据迭代器和特征维度
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
# 使用 RMSProp 优化器训练线性模型，学习率为 0.01，gamma=0.9
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim)

# ---------- 简洁实现 ----------
# trainer = torch.optim.RMSprop
# d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9}, data_iter)

# ---------- 可视化 gamma 值下的权重分布（泄漏平均） ----------
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = torch.arange(40).detach().numpy()
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend()
d2l.plt.show()
