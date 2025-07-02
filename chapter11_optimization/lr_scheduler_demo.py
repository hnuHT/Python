import math
import torch
from torch import nn
from torch.optim import lr_scheduler
from d2l import torch as d2l
import matplotlib.pyplot as plt

# ------------------------------
# 定义LeNet网络结构（使用ReLU和MaxPooling）
# ------------------------------
def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10)
    )
    return model

# ------------------------------
# 通用训练函数，支持自定义或内置调度器
# ------------------------------
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # 累积训练损失、训练准确率、样本数
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])

            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter), (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

        # 更新学习率（可选）
        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                scheduler.step()  # PyTorch内置调度器
            else:
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)  # 自定义调度器

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    plt.show()  # 显示动画曲线图

# ------------------------------
# 数据准备与训练参数设置
# ------------------------------
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()
num_epochs = 30

# ------------------------------
#（1）固定学习率训练（默认方式）
# ------------------------------
lr = 0.3
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)

# ------------------------------
#（2）平方根衰减调度器（自定义）【可解注运行】
# ------------------------------
# class SquareRootScheduler:
#     def __init__(self, lr=0.1):
#         self.lr = lr
#     def __call__(self, epoch):
#         return self.lr * pow(epoch + 1.0, -0.5)
#
# scheduler = SquareRootScheduler(lr=0.1)
# d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
# plt.show()
# net = net_fn()
# trainer = torch.optim.SGD(net.parameters(), lr=0.1)
# train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler)

# ------------------------------
#（3）FactorScheduler：指数衰减（自定义）
# ------------------------------
# class FactorScheduler:
#     def __init__(self, factor=0.9, stop_factor_lr=1e-2, base_lr=2.0):
#         self.factor = factor
#         self.stop_factor_lr = stop_factor_lr
#         self.base_lr = base_lr
#     def __call__(self, epoch):
#         self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
#         return self.base_lr
#
# scheduler = FactorScheduler()
# d2l.plot(torch.arange(50), [scheduler(t) for t in range(50)])
# plt.show()

# ------------------------------
#（4）MultiStepLR：多段恒定值（PyTorch 内置）
# ------------------------------
# net = net_fn()
# trainer = torch.optim.SGD(net.parameters(), lr=0.5)
# scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)
# def get_lr(trainer, scheduler):
#     lr = scheduler.get_last_lr()[0]
#     trainer.step()
#     scheduler.step()
#     return lr
# d2l.plot(torch.arange(num_epochs), [get_lr(trainer, scheduler) for _ in range(num_epochs)])
# plt.show()
# train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler)

# ------------------------------
# 余弦调度器 + 预热（Warmup，自定义）
# ------------------------------
# class CosineScheduler:
#     def __init__(self, max_update, base_lr=0.3, final_lr=0.01, warmup_steps=5, warmup_begin_lr=0):
#         self.base_lr_orig = base_lr
#         self.final_lr = final_lr
#         self.warmup_steps = warmup_steps
#         self.warmup_begin_lr = warmup_begin_lr
#         self.max_steps = max_update - warmup_steps
#     def get_warmup_lr(self, epoch):
#         increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch) / self.warmup_steps
#         return self.warmup_begin_lr + increase
#     def __call__(self, epoch):
#         if epoch < self.warmup_steps:
#             return self.get_warmup_lr(epoch)
#         elif epoch <= self.warmup_steps + self.max_steps:
#             return self.final_lr + (self.base_lr_orig - self.final_lr) * \
#                    (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
#         else:
#             return self.final_lr
#
# scheduler = CosineScheduler(max_update=30, base_lr=0.3, final_lr=0.01, warmup_steps=5)
# d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
# plt.show()
# net = net_fn()
# trainer = torch.optim.SGD(net.parameters(), lr=0.3)
# train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler)
