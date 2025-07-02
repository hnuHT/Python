import torch
from torch import nn
from d2l import torch as d2l  #

# --------------------------
# 定义二维互相关运算（非标准卷积，不含翻转）
def corr2d(X, K):
    """计算二维互相关运算（correlation）"""
    h, w = K.shape  # 获取核的高和宽
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))  # 输出大小取决于输入和核大小
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 对输入X的每个滑动窗口区域与核K做点乘并求和
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

# --------------------------
# 自定义一个最简单的卷积层（权重 + 偏置，forward 中使用自定义 corr2d）
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))  # 卷积核权重（可训练参数）
        self.bias = nn.Parameter(torch.zeros(1))             # 偏置（初始为0）

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# --------------------------
# 构造一个测试用图像：一个 6x8 的图，2~5 列是黑（0），其他是白（1）
X = torch.ones((6, 8))
X[:, 2:6] = 0

# 构造一个检测“垂直边缘”的卷积核
K = torch.tensor([[1.0, -1.0]])

# 得到期望输出 Y，Y 的形状为 (6, 7)
Y = corr2d(X, K)

# --------------------------
# 用 nn.Conv2d 来模拟这个过程，让它**学习出这个卷积核**

# 创建一个 2D 卷积层，参数说明：
# 输入通道=1，输出通道=1，核大小=(1,2)，不加 bias（我们希望它学出 K）
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 输入输出 reshape 成 [batch_size, channels, height, width] 的形式
X = X.reshape((1, 1, 6, 8))  # [1, 1, 6, 8]
Y = Y.reshape((1, 1, 6, 7))  # [1, 1, 6, 7]，注意宽度减少1是因为 kernel_size=(1,2)

# 学习率
lr = 3e-2

# 训练 10 次
for i in range(10):
    Y_hat = conv2d(X)           # 前向传播
    l = (Y_hat - Y) ** 2        # 计算 MSE loss（逐元素平方差）
    conv2d.zero_grad()          # 梯度清零
    l.sum().backward()          # 反向传播
    conv2d.weight.data[:] -= lr * conv2d.weight.grad  # 梯度下降更新参数

    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

# 打印最终学习到的卷积核权重
print("Learned kernel:", conv2d.weight.data.reshape((1, 2)))
