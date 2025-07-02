import math
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np


# 初始化 AdaGrad 的状态变量，每个参数对应一个累加平方梯度的变量
# s_w 和 s_b 的形状分别对应模型的权重和偏置

def init_adagrad_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))  # 权重的累积平方梯度
    s_b = torch.zeros(1)  # 偏置的累积平方梯度
    return (s_w, s_b)

# AdaGrad 优化器的核心更新逻辑
# 对每个参数进行按坐标缩放的学习率更新

def adagrad(params, states, hyperparams):
    eps = 1e-6  # 为避免除以0而加的常数
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)  # 累加梯度的平方
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)  # 更新参数
        p.grad.data.zero_()  # 梯度清零

# ---------------- 准备数据 ----------------

def get_data_ch11(batch_size=10, n=1500):
    d2l.DATA_HUB['airfoil'] = (
        d2l.DATA_URL + 'airfoil_self_noise.dat',
        '76e5be1548fd8222e5074cf0faae75edff8cf93f')
    data = np.genfromtxt(d2l.download('airfoil'), dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))  # 标准化
    return d2l.load_array((data[:n, :-1], data[:n, -1]), batch_size, is_train=True), data.shape[1] - 1

# 获取数据迭代器与特征维度
data_iter, feature_dim = get_data_ch11(batch_size=10)

# ---------------- 模型训练逻辑 ----------------

def train_adagrad(lr=0.1, num_epochs=2):
    d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
                   {'lr': lr}, data_iter, feature_dim, num_epochs)

# 启动训练
train_adagrad(lr=0.1, num_epochs=2)


# def train_concise_adagrad(lr=0.1, num_epochs=2):
#     net = nn.Sequential(nn.Linear(feature_dim, 1))  # 线性回归模型
#     def init_weights(m):
#         if type(m) == nn.Linear:
#             nn.init.normal_(m.weight, std=0.01)
#     net.apply(init_weights)
#
#     optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
#     loss = nn.MSELoss(reduction='none')
#     d2l.train_concise_ch11(optimizer, {'lr': lr}, data_iter, net, loss, num_epochs)


# train_concise_adagrad(lr=0.1, num_epochs=2)
