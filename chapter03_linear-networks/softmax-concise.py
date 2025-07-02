import torch
from torch import nn
from d2l import torch as d2l  # d2l库是《动手学深度学习》提供的工具集

# 设置批量大小
batch_size = 256

# 加载 Fashion-MNIST 数据集，返回训练和测试数据的迭代器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 构建模型：顺序包含两个层
# 1. nn.Flatten()：将每张 28x28 图像展平为 784 维向量
# 2. nn.Linear(784, 10)：线性变换，映射为10个类别的得分（logits）
net = nn.Sequential(
    nn.Flatten(),           # 展平输入图像
    nn.Linear(784, 10)      # 全连接层：输入 784 -> 输出 10
)

# 初始化权重：若是 Linear 层，将其权重初始化为均值为0，标准差为0.01的正态分布
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# 对 net 中的所有子模块应用权重初始化
net.apply(init_weights)

# 定义损失函数：交叉熵损失，用于多分类
# CrossEntropyLoss 会自动包含 softmax + log + NLLLoss
# 所以只需将 logits（未softmax）传入即可
loss = nn.CrossEntropyLoss(reduction='none')

# 定义优化器：随机梯度下降（SGD），学习率为 0.1
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 设置训练轮数
num_epochs = 10

# 使用 d2l 封装好的训练函数进行模型训练并评估
# 自动处理前向、反向传播、参数更新、测试集评估等步骤
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# 显示训练过程中绘制的图（损失、准确率）
d2l.plt.show()
