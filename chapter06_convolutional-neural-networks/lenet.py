import torch
from torch import nn
from d2l import torch as d2l  # 《动手学深度学习》PyTorch版工具包


# -------------------------------------------------------
# 自定义模块：用于将输入展平为1通道的28×28图像（Fashion-MNIST原始格式）
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)  # -1 表示自动推导批量大小，通道为1，图像为28×28


# -------------------------------------------------------
# 定义 LeNet 模型结构（经典的CNN结构，90年代提出）

# LeNet由两部分组成：
# 1️卷积编码器部分：卷积层 + sigmoid激活 + 平均池化
# 2️全连接分类器部分：线性层 + sigmoid激活，最后一层输出10个类别
net = nn.Sequential(
    Reshape(),  # 输入 reshape 为 (batch_size, 1, 28, 28)

    nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 输入通道=1，输出通道=6，5x5卷积核，padding=2保持尺寸
    nn.Sigmoid(),  # Sigmoid激活函数（原始LeNet结构所用）
    nn.AvgPool2d(kernel_size=2, stride=2),  # 平均池化层，窗口2x2，步幅为2，尺寸减半

    nn.Conv2d(6, 16, kernel_size=5),  # 第二个卷积层，输入通道=6，输出通道=16
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # 再次池化，尺寸继续减半

    nn.Flatten(),  # 4维张量 → 2维张量，为全连接层做准备（展平）

    nn.Linear(16 * 5 * 5, 120),  # 全连接层，输入400维（16×5×5），输出120维
    nn.Sigmoid(),

    nn.Linear(120, 84),  # 第二个全连接层
    nn.Sigmoid(),

    nn.Linear(84, 10)  # 输出层，10个类别对应Fashion-MNIST的10类
)

# -------------------------------------------------------
# 逐层打印每一层输出的形状，便于调试和理解网络的结构

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# -------------------------------------------------------
# 加载 Fashion-MNIST 数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


# -------------------------------------------------------
# 定义评估函数，在指定device上评估模型准确率（支持GPU）

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的准确率"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式，关闭dropout和BN等操作
        if not device:
            device = next(iter(net.parameters())).device  # 获取模型所在设备
    metric = d2l.Accumulator(2)  # 正确预测数量，总样本数量
    with torch.no_grad():  # 推理时不计算梯度
        for X, y in data_iter:
            if isinstance(X, list):  # 兼容BERT输入结构（此处无用）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            # accuracy(net(X), y) 得到该批预测正确数量
            metric.add(d2l.accuracy(net(X), y), y.numel())  # numel() = 样本数量
    return metric[0] / metric[1]  # 正确预测数 / 总数 = 准确率


# -------------------------------------------------------
# 模型训练函数：指定设备上执行训练 + 可视化训练过程

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """使用 GPU 训练模型"""

    # 权重初始化函数：卷积层和线性层使用 Xavier 均匀初始化
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)  # 对每个子模块执行初始化函数
    print('training on', device)
    net.to(device)  # 模型搬到GPU
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 随机梯度下降优化器
    loss = nn.CrossEntropyLoss()  # 多分类的交叉熵损失
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])  # 用于可视化训练过程
    timer, num_batches = d2l.Timer(), len(train_iter)

    for epoch in range(num_epochs):  # 迭代多个epoch
        metric = d2l.Accumulator(3)  # 累加器：训练损失，准确数，样本数
        net.train()  # 设置为训练模式
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()  # 梯度清零
            X, y = X.to(device), y.to(device)  # 数据搬到GPU
            y_hat = net(X)  # 正向传播
            l = loss(y_hat, y)  # 计算损失
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()

            # 每5个批次可视化一次
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        # 每个epoch后测试集评估
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


# -------------------------------------------------------
# 启动训练

lr, num_epochs = 0.9, 10  # 学习率和训练轮数
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


