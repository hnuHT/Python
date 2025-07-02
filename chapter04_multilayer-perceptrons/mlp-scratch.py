# 导入所需库
import torch  # PyTorch 主库
from torch import nn  # 神经网络模块，包含层、损失函数等
from d2l import torch as d2l  # 动手学深度学习（d2l）库，封装了数据、绘图、训练等工具函数

# 设置批量大小
batch_size = 256

# 加载 Fashion-MNIST 数据集，返回训练集和测试集的迭代器（按batch切分好的数据）
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 网络的结构参数
num_inputs = 784  # 输入层：28*28图像拉平为784维向量
num_outputs = 10  # 输出层：10类（T恤、裤子、包等）
num_hiddens = 256  # 隐藏层：256个神经元（超参数，可调整）

# 手动定义网络参数（不使用 nn.Sequential）
# 第一层权重参数，形状：[784, 256]，初始化为小的随机数
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
# 第一层偏置，形状：[256]，初始为0
b1 = nn.Parameter(torch.zeros(num_hiddens))

# 第二层权重参数，形状：[256, 10]，初始化为小的随机数
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
# 第二层偏置，形状：[10]，初始为0
b2 = nn.Parameter(torch.zeros(num_outputs))

# 将所有参数收集为一个列表，便于传入优化器
params = [W1, b1, W2, b2]


# 定义ReLU激活函数（逐元素max(0, x)）
def relu(X):
    a = torch.zeros_like(X)  # 创建与X形状一样的全0张量
    return torch.max(X, a)  # 每个元素取X和0中的最大值，相当于 relu(x) = max(0, x)


# 定义模型的前向传播逻辑
def net(X):
    # 将二维图像展平为一维向量，X的形状从 [batch, 1, 28, 28] → [batch, 784]
    X = X.reshape((-1, num_inputs))

    # 第一层：线性变换 + ReLU激活
    H = relu(X @ W1 + b1)  # @ 表示矩阵乘法

    # 第二层：线性输出（不加激活，因为损失函数内含softmax）
    return H @ W2 + b2


# 使用交叉熵损失函数，适用于多类分类
# 内部会自动进行 softmax 和 log 计算
loss = nn.CrossEntropyLoss()

# 设置训练超参数
num_epochs = 10  # 训练轮数
lr = 0.1  # 学习率

# 定义优化器，这里使用随机梯度下降（SGD）
updater = torch.optim.SGD(params, lr=lr)

# 使用 d2l 封装的训练函数，进行模型训练
# 会显示训练损失、测试准确率、每轮训练进度
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 显示训练过程中绘制的图像（准确率、损失曲线）
d2l.plt.show()

# 使用训练好的模型在测试集上进行预测，并显示前几个预测结果
# 每张图下方会显示预测类别标签
d2l.predict_ch3(net, test_iter)




























































