import torch
from torch import nn
from d2l import torch as d2l

# 初始化动量状态，v的维度与参数形状一致
def init_momentum_states(feature_dim):
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return (v_w, v_b)

# 从零实现动量法SGD
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            # 更新动量：v ← β * v + ∇p
            v[:] = hyperparams['momentum'] * v + p.grad
            # 更新参数：p ← p - 学习率 * v
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()

# 通用的训练函数，适用于从零实现的优化器
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 初始化线性模型参数
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1), requires_grad=True)
    b = torch.zeros((1,), requires_grad=True)
    net = lambda X: d2l.linreg(X, w, b)
    loss = d2l.squared_loss

    # 绘图器设置
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)  # 更新参数
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n / X.shape[0] / len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    d2l.plt.show()  # 显示图像
    return timer.cumsum(), animator.Y[0]

# 加载数据集（小批量空气动力噪声数据）
def get_data_ch11(batch_size=10, n=1500):
    d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                               '76e5be1548fd8222e5074cf0faae75edff8cf93f')
    data = np.genfromtxt(d2l.download('airfoil'), dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1] - 1

# 执行一次训练（从零实现）
def train_momentum(lr, momentum, num_epochs=2):
    return train_ch11(
        sgd_momentum,
        init_momentum_states(feature_dim),
        {'lr': lr, 'momentum': momentum},
        data_iter,
        feature_dim,
        num_epochs
    )

# 获取数据
import numpy as np
data_iter, feature_dim = get_data_ch11(batch_size=10)

# 调用训练（可以修改学习率和动量参数）
train_momentum(0.005, 0.9)


# def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
#     # 初始化模型
#     net = nn.Sequential(nn.Linear(5, 1))
#     def init_weights(m):
#         if type(m) == nn.Linear:
#             torch.nn.init.normal_(m.weight, std=0.01)
#     net.apply(init_weights)

#     optimizer = trainer_fn(net.parameters(), **hyperparams)
#     loss = nn.MSELoss(reduction='none')
#     animator = d2l.Animator(xlabel='epoch', ylabel='loss',
#                             xlim=[0, num_epochs], ylim=[0.22, 0.35])
#     n, timer = 0, d2l.Timer()
#     for _ in range(num_epochs):
#         for X, y in data_iter:
#             optimizer.zero_grad()
#             out = net(X)
#             y = y.reshape(out.shape)
#             l = loss(out, y)
#             l.mean().backward()
#             optimizer.step()
#             n += X.shape[0]
#             if n % 200 == 0:
#                 timer.stop()
#                 animator.add(n / X.shape[0] / len(data_iter),
#                              (d2l.evaluate_loss(net, data_iter, loss) / 2,))
#                 timer.start()
#     print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
#     d2l.plt.show()

# # 运行简洁实现
# trainer = torch.optim.SGD
# train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
