import torch
from torch import nn
from d2l import torch as d2l

# ----------------------------------------
# 手动实现二维池化操作的正向传播（max or avg）
# pool_size: 指定池化窗口大小 (height, width)
# mode: 'max' 为最大池化，'avg' 为平均池化
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size  # 池化窗口的高度和宽度
    # 创建输出张量，尺寸会变小
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    # 遍历所有可滑动位置
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 当前池化区域：X[i:i+ph, j:j+pw]
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

# ----------------------------------------
# 测试手动实现的池化操作

X = torch.tensor([[0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]])

# 最大池化，窗口为 2×2
print(pool2d(X, (2, 2)))
# 输出:
# tensor([[4., 5.],
#         [7., 8.]])

# 平均池化，窗口为 2×2
print(pool2d(X, (2, 2), 'avg'))
# 输出:
# tensor([[2., 3.],
#         [5., 6.]])

# ----------------------------------------
# 使用 PyTorch 自带 nn.MaxPool2d 验证池化操作

# 构造一个 1×1×4×4 的输入（batch=1, channel=1, height=4, width=4）
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)
# 输出:
# tensor([[[[ 0.,  1.,  2.,  3.],
#           [ 4.,  5.,  6.,  7.],
#           [ 8.,  9., 10., 11.],
#           [12., 13., 14., 15.]]]])

# -------------------------------
# 默认：池化窗口为 3×3，stride = 3
pool2d = nn.MaxPool2d(3)
print(pool2d(X))
# 输出:
# tensor([[[[10.]]]]) → 因为只覆盖了左上角一个 3×3 区域，最大值为 10

# -------------------------------
# padding = 1，stride = 2，扩大覆盖范围
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
# 输出:
# tensor([[[[ 5.,  7.],
#           [13., 15.]]]])

# -------------------------------
# 不对称窗口、填充、步幅：窗口 2×3，padding = (0,1)，stride = (2,3)
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pool2d(X))
# 输出:
# tensor([[[[ 5.,  7.],
#           [13., 15.]]]])

# ----------------------------------------
# 多通道情况测试
# 构造一个具有 2 个通道的输入张量：X 和 X+1 拼接
X = torch.cat((X, X + 1), 1)
print(X)
# 输出:
# tensor([[[[ 0.,  1.,  2.,  3.],
#           [ 4.,  5.,  6.,  7.],
#           [ 8.,  9., 10., 11.],
#           [12., 13., 14., 15.]],
#
#          [[ 1.,  2.,  3.,  4.],
#           [ 5.,  6.,  7.,  8.],
#           [ 9., 10., 11., 12.],
#           [13., 14., 15., 16.]]]])

# 对于每个通道分别做池化（MaxPool2d 会自动处理多个通道）
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
# 输出:
# tensor([[[[ 5.,  7.],
#           [13., 15.]],
#
#          [[ 6.,  8.],
#           [14., 16.]]]])

# 每个通道分别池化，不混合。输出 shape: (batch=1, channel=2, height=2, width=2)
