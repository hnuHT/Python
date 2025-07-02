import numpy as np
import torch
from d2l import torch as d2l  # 动手学深度学习库中的工具函数
from torch.utils import data  # 用于构建数据集和加载器
from torch import nn          # 提供神经网络模块，如 Linear 层、损失函数等

# ============================================
# 1. 生成合成数据 y = 2 * x1 - 3.4 * x2 + 4.2 + 噪声
# ============================================
true_w = torch.tensor([2, -3.4])   # 真实权重
true_b = 4.2                       # 真实偏置
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# features: (1000, 2)，labels: (1000, 1)

# ============================================
# 2. 构造数据加载器
# ============================================
def load_array(data_arrays, batch_size, is_train=True):
    # data_arrays 是一个元组 (features, labels)
    # 使用 *data_arrays 解包后传入 TensorDataset，即变成 TensorDataset(features, labels)
    dataset = data.TensorDataset(*data_arrays)

    # DataLoader 将 dataset 拆成小批量，每个 batch 大小为 batch_size
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# ============================================
# 3. 定义模型：一个线性层
# ============================================
net = nn.Sequential(nn.Linear(2, 1))
# Linear(2, 1)：输入维度是2，输出维度是1，相当于 y = w1*x1 + w2*x2 + b

# ============================================
# 4. 初始化参数
# ============================================
net[0].weight.data.normal_(0, 0.01)  # 权重初始化为均值0、标准差0.01的正态分布
net[0].bias.data.fill_(0)           # 偏置初始化为0

# ============================================
# 5. 定义损失函数和优化器
# ============================================
loss = nn.MSELoss()  # 均方误差损失函数，用于回归问题
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 随机梯度下降优化器，学习率为0.03

# ============================================
# 6. 模型训练
# ============================================
num_epochs = 3  # 总共训练3轮

for epoch in range(num_epochs):
    for X, y in data_iter:               # 遍历每个 mini-batch
        l = loss(net(X), y)              # 计算当前 batch 的预测误差
        trainer.zero_grad()              # 梯度清零
        l.backward()                     # 反向传播，计算梯度
        trainer.step()                   # 更新参数（w 和 b）

    # 每个 epoch 结束后，在整个训练集上评估一次总损失
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# ============================================
# 7. 查看模型训练得到的参数与真实参数的差异
# ============================================
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))

b = net[0].bias.data
print('b的估计误差：', true_b - b)
